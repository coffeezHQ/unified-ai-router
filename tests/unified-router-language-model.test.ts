import { UnifiedRouterLanguageModel } from '../src/unified-router-language-model';
import type { UnifiedRouterModelId, UnifiedRouterSettings, UnifiedRouterProviderConfig } from '../src/types/unified-router-provider';
import type { 
  LanguageModelV1Prompt, 
  LanguageModelV1Message,
  LanguageModelV1TextPart,
  LanguageModelV1CallOptions
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

    model = new UnifiedRouterLanguageModel(modelId, settings, config);
    mockFetch.mockClear();
  });

  describe('doGenerate', () => {
    it('should make a successful API call and return the response', async () => {
      const mockResponse = {
        choices: [{
          message: {
            role: 'assistant',
            content: 'Test response',
            reasoning: 'Test reasoning',
            logprobs: [{
              token: 'test',
              logprob: 0.5,
              topLogprobs: [{ token: 'test', logprob: 0.5 }]
            }],
            tool_calls: [{
              id: 'test-tool',
              type: 'function',
              function: {
                name: 'test_function',
                arguments: '{"test": true}'
              }
            }]
          },
          index: 0,
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 20
        },
        provider_metadata: { test: true }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
        headers: new Headers({ 'test-header': 'test-value' })
      });

      const result = await model.doGenerate({
        inputFormat: 'messages',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Test prompt' }] as LanguageModelV1TextPart[]
        }] as LanguageModelV1Message[]
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/chat/completions',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
            'X-Test': 'test-value'
          }),
          body: expect.stringContaining('"model":"test-model"')
        })
      );

      expect(result).toEqual({
        text: 'Test response',
        finishReason: 'stop',
        usage: {
          promptTokens: 10,
          completionTokens: 20
        },
        rawCall: expect.any(Object),
        rawResponse: expect.any(Object),
        warnings: [],
        reasoning: 'Test reasoning',
        logprobs: expect.any(Array),
        toolCalls: [{
          toolCallType: 'function',
          toolCallId: 'test-tool',
          toolName: 'test_function',
          args: '{"test": true}'
        }],
        providerMetadata: { test: true }
      });
    });

    it('should handle API errors', async () => {
      const mockError = {
        error: 'Test error',
        message: 'Test error message',
        code: 'test_error'
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve(mockError)
      });

      await expect(model.doGenerate({
        inputFormat: 'messages',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Test prompt' }] as LanguageModelV1TextPart[]
        }] as LanguageModelV1Message[]
      })).rejects.toThrow('Test error message');
    });

    it('should handle image generation', async () => {
      const mockResponse = {
        id: 'img-123',
        model: 'dall-e-3',
        data: [{
          url: 'https://example.com/image.png',
          b64_json: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
          revised_prompt: 'A serene landscape with mountains and a lake'
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 0,
          total_tokens: 10,
          cost: 0.01
        },
        provider_metadata: {
          provider: 'openai',
          model: 'dall-e-3'
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test image' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.unifiedrouter.com/v1/images/generations',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-api-key'
          }),
          body: JSON.stringify({
            model: 'test-model',
            prompt: 'Generate a test image',
            n: 1,
            size: '1024x1024',
            response_format: 'b64_json'
          })
        })
      );

      expect(result).toEqual({
        files: [{
          data: expect.any(Uint8Array),
          mimeType: 'image/png'
        }],
        usage: {
          promptTokens: 10,
          completionTokens: 0,
          totalTokens: 10,
          cost: 0.01
        },
        metadata: {
          provider: 'openai',
          model: 'dall-e-3'
        }
      });

      // Verify the image data was properly converted from base64
      const imageData = result.files![0].data;
      expect(imageData).toBeInstanceOf(Uint8Array);
      expect(imageData.length).toBeGreaterThan(0);
    });

    it('should handle image generation with URL response format', async () => {
      const mockResponse = {
        id: 'img-123',
        model: 'dall-e-3',
        data: [{
          url: 'https://example.com/image.png',
          revised_prompt: 'A serene landscape with mountains and a lake'
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 0,
          total_tokens: 10,
          cost: 0.01
        },
        provider_metadata: {
          provider: 'openai',
          model: 'dall-e-3'
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test image' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      });

      expect(result).toEqual({
        files: [{
          data: expect.any(Uint8Array),
          mimeType: 'image/png'
        }],
        usage: {
          promptTokens: 10,
          completionTokens: 0,
          totalTokens: 10,
          cost: 0.01
        },
        metadata: {
          provider: 'openai',
          model: 'dall-e-3'
        }
      });
    });

    it('should handle image generation API errors', async () => {
      const mockError = {
        error: {
          message: 'Invalid image generation parameters',
          type: 'invalid_request_error',
          code: 'invalid_parameters'
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => mockError,
      });

      await expect(model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test image' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      })).rejects.toThrow('Invalid image generation parameters');
    });

    it('should handle malformed image generation response', async () => {
      const mockResponse = {
        id: 'img-123',
        model: 'dall-e-3',
        data: [{
          // Missing both url and b64_json
          revised_prompt: 'A serene landscape with mountains and a lake'
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 0,
          total_tokens: 10,
          cost: 0.01
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await expect(model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test image' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      })).rejects.toThrow('Invalid image generation response');
    });

    it('should handle network errors during image generation', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test image' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      })).rejects.toThrow('Network error');
    });

    it('should handle video generation', async () => {
      const mockResponse = {
        id: 'vid-123',
        model: 'sora',
        data: [{
          url: 'https://example.com/video.mp4',
          b64_json: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
          revised_prompt: 'A serene landscape with mountains and a lake',
          duration: 10,
          fps: 30
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 0,
          total_tokens: 10,
          cost: 0.01
        },
        provider_metadata: {
          provider: 'openai',
          model: 'sora'
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test video' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.unifiedrouter.com/v1/videos/generations',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-api-key'
          }),
          body: JSON.stringify({
            model: 'test-model',
            prompt: 'Generate a test video',
            n: 1,
            duration: 10,
            fps: 30,
            response_format: 'b64_json'
          })
        })
      );

      expect(result).toEqual({
        files: [{
          data: expect.any(Uint8Array),
          mimeType: 'video/mp4'
        }],
        usage: {
          promptTokens: 10,
          completionTokens: 0,
          totalTokens: 10,
          cost: 0.01
        },
        metadata: {
          provider: 'openai',
          model: 'sora'
        }
      });

      // Verify the video data was properly converted from base64
      const videoData = result.files![0].data;
      expect(videoData).toBeInstanceOf(Uint8Array);
      expect(videoData.length).toBeGreaterThan(0);
    });

    it('should handle video generation with URL response format', async () => {
      const mockResponse = {
        id: 'vid-123',
        model: 'sora',
        data: [{
          url: 'https://example.com/video.mp4',
          revised_prompt: 'A serene landscape with mountains and a lake',
          duration: 10,
          fps: 30
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 0,
          total_tokens: 10,
          cost: 0.01
        },
        provider_metadata: {
          provider: 'openai',
          model: 'sora'
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test video' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      });

      expect(result).toEqual({
        files: [{
          data: expect.any(Uint8Array),
          mimeType: 'video/mp4'
        }],
        usage: {
          promptTokens: 10,
          completionTokens: 0,
          totalTokens: 10,
          cost: 0.01
        },
        metadata: {
          provider: 'openai',
          model: 'sora'
        }
      });
    });

    it('should handle video generation API errors', async () => {
      const mockError = {
        error: {
          message: 'Invalid video generation parameters',
          type: 'invalid_request_error',
          code: 'invalid_parameters'
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => mockError,
      });

      await expect(model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test video' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      })).rejects.toThrow('Invalid video generation parameters');
    });

    it('should handle malformed video generation response', async () => {
      const mockResponse = {
        id: 'vid-123',
        model: 'sora',
        data: [{
          // Missing both url and b64_json
          revised_prompt: 'A serene landscape with mountains and a lake',
          duration: 10,
          fps: 30
        }],
        usage: {
          prompt_tokens: 10,
          completion_tokens: 0,
          total_tokens: 10,
          cost: 0.01
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      await expect(model.doGenerate({
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Generate a test video' }]
        }],
        mode: { type: 'object-json' },
        inputFormat: 'messages'
      })).rejects.toThrow('Invalid video generation response');
    });
  });

  describe('doStream', () => {
    it('should handle streaming responses correctly', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Hello"}}]}\n'));
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":" World"}}]}\n'));
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"finish_reason":"stop"}]}\n'));
          controller.enqueue(new TextEncoder().encode('data: [DONE]\n'));
          controller.close();
        }
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: mockStream,
        headers: new Headers({ 'test-header': 'test-value' })
      });

      const result = await model.doStream({
        inputFormat: 'messages',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Test prompt' }] as LanguageModelV1TextPart[]
        }] as LanguageModelV1Message[]
      });

      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.test.com/chat/completions',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Authorization': 'Bearer test-api-key',
            'Content-Type': 'application/json',
            'X-Test': 'test-value'
          }),
          body: expect.stringContaining('"model":"test-model"')
        })
      );

      // Read the stream
      const reader = result.stream.getReader();
      const chunks: any[] = [];
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }

      expect(chunks).toEqual([
        { type: 'text-delta', textDelta: 'Hello' },
        { type: 'text-delta', textDelta: ' World' },
        {
          type: 'finish',
          finishReason: 'stop',
          usage: { promptTokens: 0, completionTokens: 0 },
          providerMetadata: undefined
        }
      ]);
    });

    it('should handle tool calls in streaming responses', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"tool_calls":[{"id":"tool1","type":"function","function":{"name":"test_function","arguments":"{}"}}]}}]}\n'));
          controller.enqueue(new TextEncoder().encode('data: {"choices":[{"finish_reason":"stop"}]}\n'));
          controller.close();
        }
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: mockStream,
        headers: new Headers({ 'test-header': 'test-value' })
      });

      const result = await model.doStream({
        inputFormat: 'messages',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Test prompt' }] as LanguageModelV1TextPart[]
        }] as LanguageModelV1Message[]
      });

      const reader = result.stream.getReader();
      const chunks: any[] = [];
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }

      expect(chunks).toEqual([
        {
          type: 'tool-call-delta',
          toolCallType: 'function',
          toolCallId: 'tool1',
          toolName: 'test_function',
          argsTextDelta: '{}'
        },
        {
          type: 'finish',
          finishReason: 'stop',
          usage: { promptTokens: 0, completionTokens: 0 },
          providerMetadata: undefined
        }
      ]);
    });

    it('should handle API errors in streaming', async () => {
      const mockError = {
        error: 'Test error',
        message: 'Test error message',
        code: 'test_error'
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: () => Promise.resolve(mockError)
      });

      await expect(model.doStream({
        inputFormat: 'messages',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Test prompt' }] as LanguageModelV1TextPart[]
        }] as LanguageModelV1Message[]
      })).rejects.toThrow('Test error message');
    });

    it('should handle malformed streaming data', async () => {
      const mockStream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"invalid": "data"}\n'));
          controller.close();
        }
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: mockStream,
        headers: new Headers({ 'test-header': 'test-value' })
      });

      const result = await model.doStream({
        inputFormat: 'messages',
        mode: { type: 'regular' },
        prompt: [{
          role: 'user',
          content: [{ type: 'text', text: 'Test prompt' }] as LanguageModelV1TextPart[]
        }] as LanguageModelV1Message[]
      });

      const reader = result.stream.getReader();
      const chunks: any[] = [];
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }

      // Should not throw but log error and continue
      expect(chunks).toEqual([]);
    });
  });
}); 