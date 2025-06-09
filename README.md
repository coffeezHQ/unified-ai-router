# unified-ai-router

[![Build Status](https://github.com/coffeezhq/unified-ai-router/actions/workflows/ci.yml/badge.svg)](https://github.com/coffeezhq/unified-ai-router/actions/workflows/ci.yml)
[![Test Coverage](https://codecov.io/gh/coffeezhq/unified-ai-router/branch/main/graph/badge.svg)](https://codecov.io/gh/coffeezhq/unified-ai-router)
[![npm version](https://badge.fury.io/js/unified-ai-router.svg)](https://badge.fury.io/js/unified-ai-router)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified provider package for the Vercel AI SDK, supporting seamless integration with multiple LLM APIs through a single, pluggable interface. This package is designed for use with the [`ai`](https://sdk.vercel.ai/docs) SDK.

## Features
- Unified interface for LLM providers (OpenAI, Anthropic, Cohere, Gemini, etc.)
- Easy configuration and API key management
- Supports extra body/provider options
- Compatible with Vercel AI SDK's `generateText` and `streamText`
- Image generation support
- Video generation support

## Installation

```bash
npm install unified-ai-router
```

## Usage

### Basic Text Generation

```ts
import { unifiedrouter } from 'unified-ai-router';
import { generateText } from 'ai';

const { text } = await generateText({
  model: unifiedrouter('openai/gpt-4o'),
  prompt: 'Write a vegetarian lasagna recipe for 4 people.',
});
```

### Streaming Text Generation

```ts
import { createUnifiedRouter } from 'unified-ai-router';
import { streamText } from 'ai';

const unifiedrouter = createUnifiedRouter({ apiKey: 'your-api-key' });
const model = unifiedrouter('anthropic/claude-3.7-sonnet:thinking');
await streamText({
  model,
  messages: [{ role: 'user', content: 'Hello' }],
  providerOptions: {
    unifiedrouter: {
      reasoning: {
        max_tokens: 10,
      },
    },
  },
});
```

### Image Generation

```ts
import { unifiedrouter } from 'unified-ai-router';
import { generateText } from 'ai';

const { files } = await generateText({
  model: unifiedrouter('openai/dall-e-3'),
  prompt: 'A serene landscape with mountains and a lake',
  mode: { type: 'object-json' }
});

// The generated image will be available as a Uint8Array in the files array
const imageData = files?.[0].data;
const mimeType = files?.[0].mimeType; // 'image/png'
```

### Video Generation

```ts
import { unifiedrouter } from 'unified-ai-router';
import { generateText } from 'ai';

const { files } = await generateText({
  model: unifiedrouter('openai/sora'),
  prompt: 'A serene landscape with mountains and a lake',
  mode: { type: 'object-json' }
});

// The generated video will be available as a Uint8Array in the files array
const videoData = files?.[0].data;
const mimeType = files?.[0].mimeType; // 'video/mp4'
```

### Custom Configuration

```ts
import { createUnifiedRouter } from 'unified-ai-router';

const unifiedrouter = createUnifiedRouter({
  apiKey: 'your-api-key',
  baseURL: 'https://api.your-llm-provider.com/v1',
  extraBody: { custom: 'value' },
});
```

## API

### `createUnifiedRouter(config)`
Creates a new provider instance.
- `config.apiKey` (**required**): API key for your LLM provider
- `config.baseURL` (optional): Override the base URL
- `config.extraBody` (optional): Extra body to send with every request

### `unifiedrouter(model, modelOptions?)`
Returns a model object for use with the AI SDK.
- `model`: The model name (e.g., 'openai/gpt-4o')
- `modelOptions`: Optional model-specific options (e.g., `extraBody`)

## Environment Variables

- `UNIFIEDROUTER_API_KEY`: Used by the default `unifiedrouter` instance

## License

MIT 