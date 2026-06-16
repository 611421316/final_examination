import json
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from src.crews.simulation_crew import SimulationCrew
from src.tools.interaction_tool_wrapper import _GLOBAL_INTERACTION_TOOL
from src.tools.exact_lookup_tools import lookup_user_by_id, lookup_item_by_id, _lookup_reviews_by_user_and_item_impl

def extract_json_from_output(raw_output: str) -> dict:
    """Extract and sanitize JSON from LLM raw output with regex fallback."""
    text = str(raw_output).strip()
    
    # Fix double curly braces {{ }} -> { }
    text = text.replace('{{', '{').replace('}}', '}')
    
    # Strategy 1: Try to find a JSON object containing "stars" and "review"
    match = re.search(r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try to find a JSON with "predicted_rating" and "generated_review"
    match = re.search(r'\{[^{}]*"predicted_rating"[^{}]*"generated_review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Try to extract a star rating number from free text
    star_match = re.search(r'(\d+\.?\d*)\s*(?:stars?|分|顆星)', text, re.IGNORECASE)
    rating = float(star_match.group(1)) if star_match else 4.0

    return {"stars": rating, "review": text}


def fetch_user_data(user_id: str) -> Optional[Dict[str, Any]]:
    if not user_id: return None
    try:
        if _GLOBAL_INTERACTION_TOOL is not None:
            user_info = _GLOBAL_INTERACTION_TOOL.get_user(user_id=user_id)
            if user_info:
                if isinstance(user_info, str):
                    try:
                        return json.loads(user_info)
                    except json.JSONDecodeError:
                        return None
                return user_info
    except Exception:
        pass
    
    # fallback to tool
    try:
        # CrewAI tool callable - returns string
        res = lookup_user_by_id(user_id=user_id)
        if isinstance(res, str) and res.startswith("{"):
            return json.loads(res)
    except Exception:
        pass
    return None


def fetch_item_data(item_id: str) -> Optional[Dict[str, Any]]:
    if not item_id: return None
    try:
        if _GLOBAL_INTERACTION_TOOL is not None:
            item_info = _GLOBAL_INTERACTION_TOOL.get_item(item_id=item_id)
            if item_info:
                if isinstance(item_info, str):
                    try:
                        return json.loads(item_info)
                    except json.JSONDecodeError:
                        return None
                return item_info
    except Exception:
        pass
    
    # fallback to tool
    try:
        res = lookup_item_by_id(item_id=item_id)
        if isinstance(res, str) and res.startswith("{"):
            return json.loads(res)
    except Exception:
        pass
    return None


def fetch_review_data(user_id: str, item_id: str) -> Optional[Dict[str, Any]]:
    if not user_id or not item_id: return None
    try:
        if _GLOBAL_INTERACTION_TOOL is not None:
            user_reviews = _GLOBAL_INTERACTION_TOOL.get_reviews(user_id=user_id)
            if user_reviews:
                if isinstance(user_reviews, str):
                    try:
                        user_reviews = json.loads(user_reviews)
                    except json.JSONDecodeError:
                        user_reviews = []
                if isinstance(user_reviews, list):
                    exact_results = [r for r in user_reviews if r.get("item_id") == item_id]
                    if exact_results:
                        return exact_results[0]
    except Exception:
        pass
    
    # fallback to tool impl
    try:
        res = _lookup_reviews_by_user_and_item_impl(user_id=user_id, item_id=item_id)
        if isinstance(res, str) and res.startswith("["):
            reviews = json.loads(res)
            exact = [r for r in reviews if r.get("user_id") == user_id and r.get("item_id") == item_id]
            if exact:
                return exact[0]
            if reviews:
                return reviews[0] # Return related if exact not found
    except Exception:
        pass
    return None


class InferenceState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    predicted_rating: float = 0.0
    generated_review: str = ""


class AgentSocietyServingFlow(Flow[InferenceState]):
    def __init__(self, agents_config_path: str = None, *args, **kwargs):
        # super().__init__ MUST come first — CrewAI's Flow base class is
        # Pydantic-backed and resets the instance __dict__ during init.
        # Setting custom attributes beforehand causes them to disappear.
        super().__init__(*args, **kwargs)
        self.agents_config_path = agents_config_path

    @start()
    def init_request(self):
        # 初始化階段，紀錄收到的 user_id 和 item_id
        pass

    @listen(init_request)
    def trigger_crew_inference(self):
        uid = self.state.user_id
        iid = self.state.item_id
        
        # Step 1: Python Query Layer
        user_info = fetch_user_data(uid)
        item_info = fetch_item_data(iid)
        review_info = fetch_review_data(uid, iid)
        print("user_info size:", uid,", ", len(user_info) if user_info else 0)
        print("item_info size:", iid,", ", len(item_info) if item_info else 0)
        print("review_info size:", len(review_info) if review_info else 0)
        # Step 2: Case Detection
        if user_info and item_info and review_info:
            case_type = "User + Restaurant + Review"
        elif user_info and item_info:
            case_type = "User + Restaurant"
        elif item_info:
            case_type = "Restaurant Only"
        else:
            case_type = "No User and No Restaurant"
            
        # Step 3: Feature Engineering (Compact profiles)
        compact_user = {}
        if user_info:
            compact_user = {
                "review_count": user_info.get("review_count"),
                "average_stars": user_info.get("average_stars"),
                "elite": user_info.get("elite"),
                "fans": user_info.get("fans")
            }
            
        compact_item = {}
        if item_info:
            compact_item = {
                "name": item_info.get("name"),
                "stars": item_info.get("stars"),
                "review_count": item_info.get("review_count"),
                "categories": item_info.get("categories"),
                "price_range": item_info.get("attributes", {}).get("RestaurantsPriceRange2") if isinstance(item_info.get("attributes"), dict) else None
            }
            
        compact_review = {}
        if review_info:
            compact_review = {
                "stars": review_info.get("stars"),
                "text": review_info.get("text")
            }
        
        # 定義傳遞到任務 {user_id} 與 {item_id} 變數的值
        inputs = {
            'case_type': case_type,
            'user_profile': json.dumps(compact_user, ensure_ascii=False) if compact_user else "null",
            'item_profile': json.dumps(compact_item, ensure_ascii=False) if compact_item else "null",
            'historical_review': json.dumps(compact_review, ensure_ascii=False) if compact_review else "null"
        }
        
        # 啟動並執行 Crew AI 團隊
        crew_instance = SimulationCrew()
        if self.agents_config_path:
            import yaml
            with open(self.agents_config_path, "r", encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                crew_instance.agents_config = config_data
                crew_instance.tasks_config = config_data

        result = crew_instance.crew().kickoff(inputs=inputs)
        
        # 使用多層 Regex 容錯解析 LLM 的回傳結果
        try:
            if result.pydantic:
                data = result.pydantic.model_dump()
            else:
                data = extract_json_from_output(result.raw)

            self.state.predicted_rating = float(data.get('stars', data.get('predicted_rating', 4.0)))
            self.state.generated_review = str(data.get('review', data.get('generated_review', 'Good.')))
        except Exception:
            # 最終備援：把整段 raw output 當 review 用
            self.state.predicted_rating = 4.0
            self.state.generated_review = str(result.raw)

        return self.state.model_dump()
