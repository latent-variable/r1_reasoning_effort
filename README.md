# R1 Reasoning Effort

## Overview
**R1 Reasoning Effort** is a multi-API reasoning pipeline within [open-webui](https://github.com/open-webui/open-webui) that integrates OpenAI and Ollama, enabling responses with varying levels of reasoning depth. It supports iterative refinement, allowing multiple reasoning cycles with context-aware adjustments.

The *Reasoning Effort* parameter controls how long the model spends thinking before finalizing a response. Ranging from 1 to 100, each increment forces an additional reasoning iteration, enhancing depth and coherence in the generated output.

[![Watch the demo](https://img.youtube.com/vi/9-w2yMoxQBM/0.jpg)](https://youtu.be/9-w2yMoxQBM)

This project is inspired by [r1-overthinker](https://github.com/qunash/r1-overthinker), extending its concepts to support multiple APIs and provide an improved structured reasoning pipeline.

## Features
- **Multi-API Support**: Choose between OpenAI and Ollama APIs for reasoning.
- **Configurable Reasoning Iterations**: Set effort levels to control the depth of reasoning.
- **Dynamic Response Refinement**: Iteratively refine responses with extended reasoning.
- **Customizable End Tokens & Prompts**: Define how and when reasoning phases end.
- **Logging & Error Handling**: Provides robust logging and handling for API errors.

## Usage
### Basic Example
The core pipeline is implemented in the `Pipe` class. It processes messages, determines reasoning effort, and interacts with the chosen API.

#### Configuration
Modify the `Valves` within the open-webui function after setup:
[Blog post on how to set a pipe up](https://o1-at-home.hashnode.dev/run-o1-at-home-privately-think-respond-pipe-tutorial-with-open-webui-ollama)


## License
This project is based on [r1-overthinker](https://github.com/qunash/r1-overthinker) and extends its functionality.

MIT License. See `LICENSE` for details.

