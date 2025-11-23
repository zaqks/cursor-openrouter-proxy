# Cursor OpenRouter Proxy (Python)

A high-performance async HTTP/2-enabled proxy server that enables Cursor IDE (including Composer) to use any LLM available through OpenRouter. Built with FastAPI and Python, this proxy translates OpenAI-compatible API requests to work with any model available on OpenRouter.

## Primary Use Case

This proxy enables Cursor IDE users to leverage any LLM available on OpenRouter through Cursor's interface, including the Composer. Simply point Cursor to this proxy with any key, and it will handle all the necessary translations to make your chosen model work as if it were GPT-4.

## Features

- **Async/Await**: Built with FastAPI and httpx for high-performance async operations
- **HTTP/2 Support**: Improved performance with HTTP/2
- **Dynamic Model Switching**: Change models via API endpoint without container reload
- **Full CORS Support**: Cross-origin requests enabled
- **Streaming Responses**: Real-time streaming support
- **Function Calling/Tools**: Support for OpenAI function calling
- **Automatic Format Conversion**: Seamless message format translation
- **OpenAI API Compatible**: Works with OpenAI client libraries
- **API Key Validation**: Secure key validation
- **Traefik Integration**: Ready for reverse proxy setup
- **Docker Support**: Easy containerized deployment

## Prerequisites

- Cursor Pro Subscription
- OpenRouter API key (must start with `sk-or-`)
- Docker and Docker Compose (for containerized deployment)
- OR Python 3.11+ (for local development)
- Traefik (optional, for reverse proxy)

## Quick Start with Docker Compose

1. Clone the repository

2. Configure environment:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenRouter API key and preferred model

3. Start with Docker Compose:
   ```bash
   docker-compose up -d
   ```

4. The proxy will be available at `http://localhost:9000`

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file with your configuration:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   OPENROUTER_MODEL=anthropic/claude-3-opus-20240229
   DEBUG=false
   ```

3. Run the server:
   ```bash
   python main.py
   ```

   The server will start on `http://localhost:9000`

## Configuration

The `.env` file controls your setup:

```bash
# Required - must start with 'sk-or-'
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional - defaults to openai/gpt-4o
# Must include provider prefix (e.g., openai/, anthropic/, google/)
OPENROUTER_MODEL=openai/gpt-4o

# Optional - enable debug logging
DEBUG=false
```

### Important Notes

- **API Key Format**: Your OpenRouter API key must start with `sk-or-` and be at least 32 characters long
- **Model Format**: Models must include a provider prefix separated by `/` (e.g., `anthropic/claude-3-opus-20240229`)
- Available models can be found at [OpenRouter's model list](https://openrouter.ai/models)

## Usage

### Configuring Cursor IDE

1. Open Cursor Settings
2. Navigate to the API settings
3. Set the API endpoint to:
   - Local: `http://localhost:9000/v1`
   - Remote: `https://cursor-proxy.$YOURDOMAIN/v1`
4. Keep the model selection as GPT-4o in Cursor
5. Enter any valid API key format (e.g., `sk-test-key`)

The proxy will automatically:
1. Translate Cursor's GPT-4o requests to your chosen model
2. Handle all necessary format conversions
3. Stream responses back to Cursor

### Dynamic Model Switching

Switch models without restarting using the API endpoint:

```bash
curl -X POST http://localhost:9000/v1/config \
  -H "Content-Type: application/json" \
  -d '{"model": "anthropic/claude-3-opus-20240229"}'
```

Get current configuration:

```bash
curl http://localhost:9000/v1/config
```

### Health Check

Check if the proxy is running and can connect to OpenRouter:

```bash
curl http://localhost:9000/health
```

### Traefik Integration

The included `docker-compose.yml` already has Traefik labels configured. Make sure your Traefik instance is running and the `proxy` network exists.

## Supported Endpoints

- `GET /health` - Health check endpoint
- `POST /v1/chat/completions` - Chat completions endpoint (streaming and non-streaming)
- `GET /v1/models` - Models listing endpoint (proxies to OpenRouter)
- `GET /v1/config` - Get current model configuration
- `POST /v1/config` - Update model configuration

## API Documentation

When running locally, visit `http://localhost:9000/docs` for interactive API documentation (Swagger UI) or `http://localhost:9000/redoc` for ReDoc documentation.

## Security

- CORS headers for cross-origin requests
- API key validation (format checking)
- Secure request/response handling
- HTTPS support through HTTP/2
- Environment variables protection
- Input validation for all endpoints

## Performance

The Python implementation uses:
- **FastAPI**: High-performance async web framework
- **httpx**: Modern async HTTP client with HTTP/2 support
- **Connection Pooling**: Reuses connections (max 100 connections, 20 keepalive)
- **Async/Await**: Non-blocking I/O for handling multiple requests
- **5-minute timeout**: Configurable timeout for long-running requests

## Model-Specific Features

### Temperature Handling
- **Mistral AI and Google models**: Temperature is automatically capped at 1.0
- **Other models**: Temperature passed as-is

### Provider Headers
- Mistral AI models: Includes `X-Model-Provider: mistral` header
- Google models: Includes `X-Model-Provider: google` header

## Differences from Go Version

This Python implementation maintains feature parity with the Go version while providing:
- More readable, Pythonic code
- Automatic API documentation via FastAPI
- Easier debugging with Python's ecosystem
- Type hints for better IDE support
- Similar performance for typical use cases

## Troubleshooting

### Enable Debug Logging

Set `DEBUG=true` in your `.env` file and restart the server:

```bash
DEBUG=true
```

Then restart:
```bash
# Docker
docker-compose restart

# Local
# Stop the server (Ctrl+C) and run again
python main.py
```

### Check Logs

```bash
# Docker
docker-compose logs -f cursor-proxy

# Local
# Logs will appear in your terminal
```

### Common Issues

#### "OPENROUTER_API_KEY must start with 'sk-or-'"
- Ensure your API key starts with `sk-or-`
- Get your key from [OpenRouter](https://openrouter.ai/keys)

#### "Invalid model: Must contain a provider prefix"
- Model names must include the provider (e.g., `openai/gpt-4o`, not just `gpt-4o`)
- Check [OpenRouter's model list](https://openrouter.ai/models) for valid model names

#### Connection Issues
- Verify the proxy is running: `curl http://localhost:9000/health`
- Check firewall settings
- Ensure port 9000 is not in use by another service

### Test the Proxy

```bash
# Health check
curl http://localhost:9000/health

# Get available models
curl http://localhost:9000/v1/models

# Test chat completion
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-test-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | Your OpenRouter API key (must start with `sk-or-`) |
| `OPENROUTER_MODEL` | No | `openai/gpt-4o` | Default model to use (must include provider prefix) |
| `DEBUG` | No | `false` | Enable debug logging |

## License

This project is licensed under the GNU General Public License v2.0 (GPLv2). See the [LICENSE.md](LICENSE.md) file for details.

## Contributing

Contributions are welcome! Please ensure your code:
- Follows Python best practices (PEP 8)
- Includes type hints
- Works with Python 3.11+
- Passes basic testing
- Updates documentation as needed

## Credits

This is a Python rewrite of the original Go-based Cursor OpenRouter Proxy.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the [OpenRouter documentation](https://openrouter.ai/docs)
- Review the built-in API docs at `/docs` when running locally