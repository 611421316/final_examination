**🏆 \[Final Project Guidelines\]**   
**Evolutionary Multi-Agent System for AgentSociety Challenge**

**Project Overview**

This course’s final project will tackle the **AgentSociety Challenge (Track 1: User Modeling)** at the top-tier academic conference, WWW 2025\.

Your goal is to develop an AI system capable of accurately simulating real human behavior. This endeavor goes beyond merely calling an LLM API. Throughout this semester, we will integrate three core technologies you have learned:

1. **CrewAI**: Establish a multi-agent collaboration architecture (e.g., a dialogue between an analyst and a critic).  
2. **Retrieval-Augmented Generation (RAG)**: Utilize vector retrieval to extract historical user memory.  
3. **OpenEvolve**: Implement Meta-LLM-driven genetic mutation, allowing the Agent to self-evolve within the evaluation loop.

**🗓️ Milestones & Deliverables**

The project is structured around four key checkpoints aligned with the semester schedule. All teams must keep pace to ensure sufficient time for model optimization before the final week.

**Milestone 1: Static Baseline System (Midterm Exam, Week 9\)**

* **Objective**: Complete the basic encapsulation of the CrewAIWrapperAgent and successfully run it once on the official Simulator.  
* **Deliverables**:  
  1. The baseline\_agent.py source code.  
  2. A Terminal screenshot showing the execution of the official evaluation script on the Mini-Dataset (e.g., 50 entries), clearly displaying the initial **MAE (Star Rating Error)** and **Text Error**.  
* **Key Requirement**: The Agent Prompt for this stage can be manually designed, with the primary focus being a smooth data flow (Simulator \-\> CrewAI \-\> Simulator).

**Milestone 2: Evolutionary Engine Activation (Week 15/16)**

* **Objective**: Successfully integrate OpenEvolve into your Wrapper Agent and define an effective **Fitness Function**.  
* **Deliverables**:  
  1. The source code includes the evaluate\_fitness function.  
  2. Log records from OpenEvolve executing at least 3 Generations, demonstrating the system’s mechanism for automatically mutating Prompts and re-evaluating scores.

**🚀 Final Project Sprint: System Acceleration and Optimization (Weeks 16-17)**

As the semester concludes, all teams will enter a two-week sprint period, with class time reserved for group implementation, discussion, and guidance. Each week has a suggested optimization theme:

**Sprint 1 (Week 16): Scaling Up and Evolutionary Stability**[1](https://docs.google.com/document/d/180Eo1tHgZab_ZwPhQi_Wq8a3MTKuGH5zF0o5BYJmzvs/edit)

* **Task**: Expand the evaluation dataset and observe the convergence behavior of OpenEvolve on a larger set. Resolve potential issues like **data format drift (JSON Parse Error)** or **LLM hallucinations** that may occur during evolution.  
* **Output**: Confirm that the evolutionary loop runs stably for over 10 generations.  
* Begin drafting the final technical report.

**Sprint 2 (Week 17): Enhanced Memory Retrieval (Vector Search Optimization)**

* **Task**: Relying solely on the Prompt may be insufficient to simulate complex users. This week focuses on optimizing the Agent's internal RAG mechanism. Your code must accurately retrieve historical reviews from the massive database that reflect the user's **liveliness, humanity, and specific preferences**, serving as a dynamic Context for CrewAI.  
* **Output**: Demonstrate how the retrieval strategy impacts the overall Fitness Score.  
* Finalize the State-of-the-Art (SOTA) architecture and Prompt "genes," and continue writing the final report.

**📦 Final Deliverables (Week 18\)**

Please submit the following three outputs on the day of the final exam, and prepare for a 15-minute Group Presentation (Launch & Conclusion):

1. **GitHub Repository (Codebase)**  
   * Contain complete, executable source code.[1](https://docs.google.com/document/d/180Eo1tHgZab_ZwPhQi_Wq8a3MTKuGH5zF0o5BYJmzvs/edit)  
   * Include a clear README.md explaining how to set up the environment and run your evolutionary script.  
2. **Technical Report**  
   * **Architecture Diagram**: Illustrate the system’s data flow and Agent interaction mechanism.  
   * **Evolutionary Analysis**: Compare the content differences between Gen-0 (handwritten Prompt) and the final generation (evolved Prompt), discussing what unexpected strategies OpenEvolve discovered.  
   * **Performance Metrics**: Clearly list the final scores for official evaluation metrics such as MAE and Emotional Tone.  
3. **Class Presentation**  
   * Focus on sharing the biggest technical bottlenecks encountered during your Sprint (e.g., API Rate Limits, Agent infinite loops) and how you resolved them.

**📊 Grading (30% of Course Grade)**

The final project grading considers not only the final score but also the completeness of the system engineering and the depth of analysis:[1](https://docs.google.com/document/d/180Eo1tHgZab_ZwPhQi_Wq8a3MTKuGH5zF0o5BYJmzvs/edit)

* **Evaluation Results (20%)**  
* **Experimental Analysis (10%)**:  
  * E.g., Did ablation studies prove the effectiveness of individual modules? Was the insight into the evolution process profound?

**We will accomplish this challenge together. Good Luck\!**