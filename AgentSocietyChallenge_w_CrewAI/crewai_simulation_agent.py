import sys
import os

# Append current working dir to sys path so absolute imports like 'src.flows...' work seamlessly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from websocietysimulator.agent import SimulationAgent
from src.flows.serving_flow import AgentSocietyServingFlow, InferenceState
from src.tools.interaction_tool_wrapper import inject_simulator_tool

class CrewAISimulationAgent(SimulationAgent):
    """
    Adapter connecting AgentSociety's simulator framework to the CrewAI flow.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def workflow(self):
        # 1. 解析官方 Simulator 給予的任務與上下文
        # 假設官方會把目前的場景資訊塞進 self.task，且為以字典形式操作的特徵
        current_user_id = self.task.get('user_id', '') if isinstance(self.task, dict) else getattr(self.task, 'user_id', '')
        current_item_id = self.task.get('item_id', '') if isinstance(self.task, dict) else getattr(self.task, 'item_id', '')

        # 2. 將官方提供的 interaction_tool 動態注入到全局模組，令 CrewAI Tool Wrapper 可以查資料
        inject_simulator_tool(getattr(self, 'interaction_tool', None))
        
        # 3. 初始化 InferenceState，並掛載給我們定義好的 CrewAI Serving Flow
        initial_state = InferenceState(
            user_id=current_user_id,
            item_id=current_item_id
        )
        
        # 4. 實例化並觸發 CrewAI 引擎非同步、無縫執行
        flow = AgentSocietyServingFlow(initial_state=initial_state)
        final_state_dict = flow.kickoff()
        
        # 5. 按照 AgentSociety Track 1 要求，回傳 dictionary
        return {
            'stars': float(final_state_dict.get('predicted_rating', 4.0)),
            'review': str(final_state_dict.get('generated_review', 'Good.'))
        }
