import os

with open('config/evolve_bundle_6files.py', 'w') as f:
    # 1. Base (agents, tasks, crew)
    with open('config/evolve_bundle.py', 'r') as b:
        base_content = b.read()
    
    # We want to insert the new sections before EVOLVE-BLOCK-END
    end_marker = "# EVOLVE-BLOCK-END"
    if end_marker in base_content:
        pre, post = base_content.split(end_marker)
        f.write(pre)
        
        f.write("\n\n# === SECTION: lookup ===\n")
        with open('src/tools/exact_lookup_tools.py', 'r') as src:
            f.write(src.read())
            
        f.write("\n\n# === SECTION: flow ===\n")
        with open('src/flows/serving_flow.py', 'r') as src:
            f.write(src.read())

        f.write("\n\n# === SECTION: interaction ===\n")
        with open('src/tools/interaction_tool_wrapper.py', 'r') as src:
            f.write(src.read())
            
        f.write("\n" + end_marker + post)
    else:
        print("EVOLVE-BLOCK-END not found")
