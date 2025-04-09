
# Hippycampus

A LangChain-based CLI and MCP server that supports dynamic loading of OpenAPI specifications and integration with Langflow.

## Prerequisites

- Python 3.12.9
- UV package manager
- Google AI Studio API key
- Langflow (for visual workflow creation)

## Installation

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install hippycampus and its dependencies
uv pip install -e .

# Install langflow
uv pip install langflow
```

## Configuration

### Google AI Studio API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key" in the top right
3. Copy the generated key and set it as an environment variable:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

## Running the Applications

### CLI Mode (no MCP server)

```bash
uv run hippycampus-cli
```

### MCP Server Mode (SSE)

```bash
uv run hippycampus-server --transport sse --port 8000
```

### Langflow Server

Ensure the MCP server is running before starting Langflow.

1. Set the components path environment variable:
```bash
# Get your current working directory
pwd
# Use the output to set the components path
export LANGFLOW_COMPONENTS_PATH="/output/from/pwd/langflow/components"
```

2. Start the Langflow server (add --dev for development mode):
```bash
uv run langflow run
```

3. Open your browser and navigate to `http://localhost:7860`

### Using Custom Components in Langflow

1. In the Langflow UI, locate the custom components:
   - OpenApi Service: For loading OpenAPI specifications
   - Hippycampus MCP Server: For connecting to the MCP server over SSE

2. Configure the components:
   - OpenApi Service: Use `https://raw.githubusercontent.com/APIs-guru/unofficial_openapi_specs/master/xkcd.com/1.0.0/openapi.yaml` for testing
   - MCP Server: Use `http://localhost:8000/sse`

See the Screencast Demo for a visual guide.
[Screencast Demo](https://www.youtube.com/watch?v=ki5A7PvRQwQ)

**Note that the official XKCD swagger files contain an error and specify the comic_id field as a number instead of an
integer, there is a fixed version in the test folder.**

## Troubleshooting

- Authentication errors: Check if `GOOGLE_API_KEY` is set correctly
- Missing components in Langflow: Verify `LANGFLOW_COMPONENTS_PATH` points to the correct directory
- Connection issues: Ensure the MCP server is running before connecting via Langflow
- If components don't appear in Langflow, try restarting the Langflow server
- Use the cli to debug openapi_builder/spec_parser and agent interaction issues before running in MCP/Langflow.

## License

MIT License

Copyright (c) 2024 Ray Cromwell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
