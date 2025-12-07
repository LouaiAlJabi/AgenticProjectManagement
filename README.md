# Agentic Project Management

**Project Overview**

This project implements a small AI agent that takes a natural-language description of a task (for example, “Write my research paper for AI class”) and automatically:

* Decomposes the task into microsteps using a Gemini large language model.

* Builds a directed flowchart representing those microsteps and their transitions.

* Visualizes the flowchart using NetworkX and Matplotlib.

* Provides a simple desktop GUI where users can type their task and generate the flowchart with one button click.

The result is both an on-screen flowchart and a PNG image file saved to your current working directory. This is intended as a productivity and planning tool to help students, researchers, and other users break down large tasks into manageable, ordered microsteps.

**Main Components (Code Structure)**

All logic is in a single file, AlJabi_CS574_AIAgent.py, which defines:
* **MicrostepGenerator**
  Wraps the Gemini API (google.generativeai). It:
  * Builds a structured prompt that instructs the model to output:
    * Microsteps: [(id, label, type), ...]
    * Transitions: [(source_id, target_id, label), ...]
  * Calls the model with the task and returns the raw text output.
* **parse_response(test)**
  * Uses regular expressions and a custom tuple parser to extract:
    * A list of microsteps (id, label, type)
    * A list of transitions (source_id, target_id, label)
  * Inserts a \n within labels to make multi-line node labels and improve readability.
* **DrawFlowchart**
  * Builds a directed graph in NetworkX from microsteps and transitions.
  * Applies a custom left-to-right layered layout that uses:
    * Topological generations to form layers.
    * A “zigzag” vertical offset so the flowchart is not one cramped horizontal line.
  * Draws nodes with different shapes and colors based on type:
    * start, process, decision, end
  * Draws directed edges and edge labels using Matplotlib.
  * Saves the figure as microsteps_flowchart.png in the current directory and shows      it.
* **create_flowchart(task)**
  * High-level pipeline:
    * Calls MicrostepGenerator().createSteps(task)
    * Parses the LLM response via parse_response.
    * Instantiates DrawFlowchart and calls draw(...).
* **FlowchartCreator()**
  * Tkinter GUI launcher:
    * Multi-line text box to type the task.
    * “Generate Flowchart” button.
    * Informational greeting that explains what the tool does.
  * On button click: calls create_flowchart(user_text) and shows errors via message      boxes.
* The if __name__ == "__main__": block simply runs FlowchartCreator().

**Prerequisites**
* Python: 3.9+ recommended
* Python packages:

```
pip install google-generativeai networkx matplotlib
```
* Note: tkinter is natively part of the standard Python library, but on some Linux systems you may need to install it separately (try, sudo apt-get install python3-tk).

* Google Gemini API key:
  * Create an API key via Google AI Studio / Gemini console. (https://aistudio.google.com/) 
  * In the code you will see a call to:
    ```
    genai.configure(api_key="YOUR_API_KEY_HERE")
    ```
  * Replace the placeholder with your own key

**Installation**
* Clone or copy the project into a local folder:
```
git clone github.com/LouaiAlJabi/AgenticProjectManagement
cd <your-directory>
```
* Install dependencies:
```
pip install google-generativeai networkx matplotlib
```
* Configure the API key:
  * Edit AlJabi_CS574_AIAgent.py line 8 and insert your Gemini API key

**How to run**
From the directory containing AlJabi_CS574_AIAgent.py:
```
python AlJabi_CS574_AIAgent.py
```
This will:

1. Open a small window titled “Flowchart Launcher”.
2. Show a greeting explaining that the system will generate a flowchart of microsteps for your task.
3. Provide a multi-line text box labeled “Please Enter Your Task:”.

To use:
1. Type a reasonably detailed description of your task, for example:
  _Plan and write a 10-page research paper on human–AI interaction for my CS class._
2. Click “Generate Flowchart”.
3. The agent will:
  * Call the Gemini model to generate microsteps and transitions.
  * Parse the model output.
  * Build and display a flowchart in a Matplotlib window.
  * Save the flowchart as microsteps_flowchart.png in the current directory.

**Note:** If you click the button multiple times without renaming the PNG, the file will be overwritten. Rename the PNG if you wish to keep it.

**Troubleshooting**

* **Empty input warning:**
  * If you click the button without entering text, the GUI shows a warning and does     nothing.
* **LLM or parsing errors:**
  * If the model does not obey the requested _Microsteps: [...] Transitions: [...]_       format, parse_response may raise a ValueError.
  * In this case you will see an error message box.
  * Try rephrasing the task more clearly or rerunning (LLM outputs can vary).
* **API / network errors:**
  * Incorrect key, quota issues, or connectivity problems will raise exceptions         inside createSteps, surfaced in the GUI as error dialogs.
