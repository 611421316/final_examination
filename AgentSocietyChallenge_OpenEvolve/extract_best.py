import re

def _extract_section(content: str, section_name: str) -> str:
    pattern = (
        r"# === SECTION: " + re.escape(section_name) + r" ===\n"
        r"(.*?)"
        r"(?=# === SECTION:|# EVOLVE-BLOCK-END|\Z)"
    )
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip() + "\n"

with open('config/openevolve_output_6files/checkpoints/checkpoint_5/best_program.py', 'r') as f:
    content = f.read()

# 1. Agents
agents_text = _extract_section(content, "agents")
if agents_text:
    with open('config/agents_evolving.yaml', 'w') as f:
        f.write(agents_text)
    print("Extracted agents_evolving.yaml")

# 2. Tasks
tasks_text = _extract_section(content, "tasks")
if tasks_text:
    with open('config/tasks_evolving.yaml', 'w') as f:
        f.write(tasks_text)
    print("Extracted tasks_evolving.yaml")

# 3. Crew
crew_text = _extract_section(content, "crew")
if crew_text:
    crew_match = re.search(r"CREW_CONFIG\s*=\s*(\{.*?\})\s*", crew_text, re.DOTALL | re.MULTILINE)
    if crew_match:
        crew_literal = "CREW_CONFIG = " + crew_match.group(1)
        with open('src/crews/simulation_crew.py', 'r') as f:
            sc = f.read()
        sc_new = re.sub(r"CREW_CONFIG\s*=\s*\{.*?\}", crew_literal, sc, flags=re.DOTALL)
        with open('src/crews/simulation_crew.py', 'w') as f:
            f.write(sc_new)
        print("Extracted simulation_crew.py (CREW_CONFIG)")

# 4. Lookup
lookup_text = _extract_section(content, "lookup")
if lookup_text:
    with open('src/tools/exact_lookup_tools.py', 'w') as f:
        f.write(lookup_text)
    print("Extracted exact_lookup_tools.py")

# 5. Flow
flow_text = _extract_section(content, "flow")
if flow_text:
    with open('src/flows/serving_flow.py', 'w') as f:
        f.write(flow_text)
    print("Extracted serving_flow.py")

# 6. Interaction
interaction_text = _extract_section(content, "interaction")
if interaction_text:
    with open('src/tools/interaction_tool_wrapper.py', 'w') as f:
        f.write(interaction_text)
    print("Extracted interaction_tool_wrapper.py")

