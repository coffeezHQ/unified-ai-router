import { UnifiedRouterLanguageModel } from '../src/unified-router-language-model';
import type { UnifiedRouterModelId, UnifiedRouterSettings, UnifiedRouterProviderConfig } from '../src/types/unified-router-provider';
import type { 
  LanguageModelV1Prompt, 
  LanguageModelV1Message,
  LanguageModelV1TextPart,
  LanguageModelV1CallOptions,
  LanguageModelV1StreamPart,
  LanguageModelV1FinishReason,
  LanguageModelV1ProviderMetadata,
} from '@ai-sdk/provider';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('UnifiedRouterLanguageModel', () => {
  let model: UnifiedRouterLanguageModel;
  let settings: UnifiedRouterSettings;
  let config: UnifiedRouterProviderConfig;
  const modelId: UnifiedRouterModelId = 'test-model';

  beforeEach(() => {
    settings = {
      apiKey: 'test-api-key',
      baseURL: 'https://api.test.com',
      extraBody: { test: true },
      headers: { 'X-Test': 'test-value' }
    };

    config = {
      provider: 'unified-ai-router',
      compatibility: 'compatible',
      headers: () => ({
        'Authorization': `Bearer ${settings.apiKey}`,
        'Content-Type': 'application/json',
        ...settings.headers
      }),
      url: ({ modelId, path }) => `${settings.baseURL}${path}`,
      extraBody: settings.extraBody,
    };

    model = new UnifiedRouterLanguageModel({ provider: 'unified-ai-router', model: modelId, apiKey: 'test-api-key', baseURL: 'https://api.test.com' });
    mockFetch.mockClear();
  });

  describe('doGenerate', () => {
    it('should make a successful API call', async () => {
      const mockResponse = {
        choices: [{
          message: {
            content: 'Test response',
          },
          finish_reason: 'stop',
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20,
          total_tokens: 30,
        },
        provider_metadata: {
          provider: 'test-provider',
          model: 'test-model',
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
        headers: new Headers({
          'X-Test': 'test-value',
        }),
      });

      const result = await model.doGenerate({
        inputFormat: 'prompt',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{
            type: 'text',
            text: 'Test prompt',
          }],
        }] as LanguageModelV1Message[],
      });

      expect(result.text).toBe('Test response');
      expect(result.finishReason).toBe('stop');
      expect(result.usage).toEqual({
        promptTokens: 10,
        completionTokens: 20,
        totalTokens: 30,
      });
      expect(result.providerMetadata).toEqual({
        provider: 'test-provider',
        model: 'test-model',
      });
      expect(result.rawResponse?.headers).toEqual({
        'x-test': 'test-value',
      });
    });

    it('should handle API errors', async () => {
      const mockError = {
        error: {
          message: 'Test error message',
          type: 'invalid_request_error',
          code: 'invalid_api_key',
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve(mockError),
        headers: new Headers({
          'X-Test': 'test-value',
        }),
      });

      await expect(model.doGenerate({
        inputFormat: 'messages',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{
            type: 'text',
            text: 'Test prompt',
          }],
        }] as LanguageModelV1Message[],
      })).rejects.toThrow('Test error message');
    });

    it('should handle image generation', async () => {
      const mockResponse = {
        data: [{
          b64_json: 'test-base64-data',
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 0,
          total_tokens: 10,
        },
        provider_metadata: {
          provider: 'test-provider',
          model: 'test-image-model',
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
        headers: new Headers({
          'X-Test': 'test-value',
        }),
      });

      const result = await model.doGenerate({
        inputFormat: 'messages',
        mode: { type: 'object-json' },
        prompt: [{
          role: 'user',
          content: [{
            type: 'text',
            text: 'Test prompt',
          }],
        }] as LanguageModelV1Message[],
      });

      expect(result.files?.[0].mimeType).toBe('image/png');
      expect(result.files?.[0].data).toBeInstanceOf(Uint8Array);
      expect(result.usage).toEqual({
        promptTokens: 10,
        completionTokens: 0,
        totalTokens: 10,
      });
      expect(result.providerMetadata).toEqual({
        provider: 'test-provider',
        model: 'test-image-model',
      });
      expect(result.rawResponse?.headers).toEqual({
        'x-test': 'test-value',
      });
    });

    it('should handle video generation', async () => {
      const mockResponse = {
        data: [{
          b64_json: 'test-base64-data',
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 0,
          total_tokens: 10,
        },
        provider_metadata: {
          provider: 'test-provider',
          model: 'test-video-model',
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
        headers: new Headers({
          'X-Test': 'test-value',
        }),
      });

      // Instantiate the model with 'test-video-model' for this test
      model = new UnifiedRouterLanguageModel({ provider: 'unified-ai-router', model: 'test-video-model', apiKey: 'test-api-key', baseURL: 'https://api.test.com' });

      const result = await model.doGenerate({
        inputFormat: 'messages',
        mode: { type: 'object-json' },
        prompt: [{
          role: 'user',
          content: [{
            type: 'text',
            text: 'Test prompt',
          }],
        }] as LanguageModelV1Message[],
      });

      expect(result.files?.[0].mimeType).toBe('video/mp4');
      expect(result.files?.[0].data).toBeInstanceOf(Uint8Array);
      expect(result.usage).toEqual({
        promptTokens: 10,
        completionTokens: 0,
        totalTokens: 10,
      });
      expect(result.providerMetadata).toEqual({
        provider: 'test-provider',
        model: 'test-video-model',
      });
      expect(result.rawResponse?.headers).toEqual({
        'x-test': 'test-value',
      });
    });
  });

  describe('doStream', () => {
    it('should handle streaming responses correctly', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'));
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":" world"}}]}\n\n'));
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"finish_reason":"stop"}]}\n\n'));
          controller.close();
        },
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: mockStream,
        headers: new Headers({
          'X-Test': 'test-value',
        }),
      });

      const result = await model.doStream({
        inputFormat: 'prompt',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{
            type: 'text',
            text: 'Test prompt',
          }],
        }] as LanguageModelV1Message[],
      });

      const chunks: LanguageModelV1StreamPart[] = [];
      const reader = result.stream.getReader();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }

      expect(chunks).toEqual([
        { type: 'text-delta', textDelta: 'Hello' },
        { type: 'text-delta', textDelta: ' world' },
        expect.objectContaining({
          type: 'finish',
          finishReason: 'stop',
        }),
      ]);
      expect(result.rawResponse?.headers).toEqual({
        'x-test': 'test-value',
      });
    });

    it('should handle tool calls in streaming responses', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"tool_calls":[{"id":"call_123","function":{"name":"test_function","arguments":"{}"}}]}}]}\n\n'));
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"finish_reason":"tool_calls"}]}\n\n'));
          controller.close();
        },
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: mockStream,
        headers: new Headers({
          'X-Test': 'test-value',
        }),
      });

      const result = await model.doStream({
        inputFormat: 'prompt',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{
            type: 'text',
            text: 'Test prompt',
          }],
        }] as LanguageModelV1Message[],
      });

      const chunks: LanguageModelV1StreamPart[] = [];
      const reader = result.stream.getReader();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }

      expect(chunks).toEqual([
        {
          type: 'tool-call-delta',
          toolCallType: 'function',
          toolCallId: 'call_123',
          toolName: 'test_function',
          argsTextDelta: '{}',
        },
        expect.objectContaining({
          type: 'finish',
          finishReason: 'tool_calls',
        }),
      ]);
      expect(result.rawResponse?.headers).toEqual({
        'x-test': 'test-value',
      });
    });

    it('should handle API errors', async () => {
      const mockError = {
        error: {
          message: 'Test error message',
          type: 'invalid_request_error',
          code: 'invalid_api_key',
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve(mockError),
        headers: new Headers({
          'X-Test': 'test-value',
        }),
      });

      await expect(model.doStream({
        inputFormat: 'messages',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{
            type: 'text',
            text: 'Test prompt',
          }],
        }] as LanguageModelV1Message[],
      })).rejects.toThrow('Test error message');
    });
  });
}); 