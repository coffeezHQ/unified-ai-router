import type { 
  LanguageModelV1, 
  LanguageModelV1FinishReason, 
  LanguageModelV1CallOptions, 
  LanguageModelV1StreamPart, 
  LanguageModelV1ProviderMetadata,
  LanguageModelV1FunctionToolCall,
  LanguageModelV1LogProbs,
  LanguageModelV1ImagePart
} from '@ai-sdk/provider';
import { createJsonResponseHandler, postJsonToApi, createJsonErrorResponseHandler, createEventSourceResponseHandler } from '@ai-sdk/provider-utils';
import { UnifiedRouterChatCompletionResponseSchema } from './schemas/unified-router-chat-completion';
import { z } from 'zod';
import type {
  UnifiedRouterModelId,
  UnifiedRouterSettings,
  UnifiedRouterProviderConfig,
} from './types/unified-router-provider';

const UnifiedRouterErrorSchema = z.object({ 
  error: z.string().optional(),
  code: z.string().optional(),
  type: z.string().optional(),
  message: z.string().optional()
});

// Enhanced streaming chunk schema with tool calls and metadata
const UnifiedRouterStreamChunkSchema = z.object({
  choices: z.array(
    z.object({
      delta: z.object({ 
        content: z.string().optional(),
        tool_calls: z.array(z.object({
          id: z.string(),
          type: z.string(),
          function: z.object({
            name: z.string(),
            arguments: z.string()
          })
        })).optional(),
        reasoning: z.string().optional(),
        logprobs: z.array(z.object({
          token: z.string(),
          logprob: z.number(),
          topLogprobs: z.array(z.object({
            token: z.string(),
            logprob: z.number()
          }))
        })).optional()
      }).optional(),
      finish_reason: z.string().optional().nullable(),
      index: z.number().optional(),
    })
  ),
  provider_metadata: z.record(z.unknown()).optional()
});

// Add image generation response schema
const UnifiedRouterImageGenerationResponseSchema = z.object({
  id: z.string().optional(),
  model: z.string().optional(),
  data: z.array(
    z.object({
      url: z.string().optional(),
      b64_json: z.string().optional(),
      revised_prompt: z.string().optional()
    })
  ),
  usage: z
    .object({
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
      total_tokens: z.number().optional(),
      cost: z.number().optional(),
    })
    .optional(),
  provider_metadata: z.record(z.unknown()).optional()
});

// Add video generation response schema
const UnifiedRouterVideoGenerationResponseSchema = z.object({
  id: z.string().optional(),
  model: z.string().optional(),
  data: z.array(
    z.object({
      url: z.string().optional(),
      b64_json: z.string().optional(),
      revised_prompt: z.string().optional(),
      duration: z.number().optional(),
      fps: z.number().optional()
    })
  ),
  usage: z
    .object({
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
      total_tokens: z.number().optional(),
      cost: z.number().optional(),
    })
    .optional(),
  provider_metadata: z.record(z.unknown()).optional()
});

// Extended message type to include additional fields
interface ExtendedMessage {
  role: string;
  content?: string | null;
  reasoning?: string;
  logprobs?: Array<{
    token: string;
    logprob: number;
    topLogprobs: Array<{
      token: string;
      logprob: number;
    }>;
  }>;
  tool_calls?: Array<{
    id: string;
    type: string;
    function: {
      name: string;
      arguments: string;
    };
  }>;
}

// Extended response type to include provider metadata
interface ExtendedResponse {
  choices: Array<{
    message: ExtendedMessage;
    index: number;
    finish_reason?: string | null;
  }>;
  id?: string;
  model?: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
  };
  provider_metadata?: Record<string, unknown>;
}

export class UnifiedRouterLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1';
  readonly defaultObjectGenerationMode = 'tool';

  readonly modelId: UnifiedRouterModelId;
  readonly settings: UnifiedRouterSettings;
  private readonly config: UnifiedRouterProviderConfig;

  constructor(
    modelId: UnifiedRouterModelId,
    settings: UnifiedRouterSettings,
    config: UnifiedRouterProviderConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  get provider(): string {
    return this.config.provider;
  }

  async doGenerate(options: Parameters<LanguageModelV1['doGenerate']>[0]) {
    const args = {
      model: this.modelId,
      ...this.settings.extraBody,
      ...options,
    };

    // Handle image generation mode
    if (options.mode?.type === 'object-json' && this.modelId.includes('dall-e')) {
      const { responseHeaders, value: response } = await postJsonToApi({
        url: this.config.url({ path: '/images/generations', modelId: this.modelId }),
        headers: this.config.headers(),
        body: args,
        failedResponseHandler: createJsonErrorResponseHandler({
          errorSchema: UnifiedRouterErrorSchema,
          errorToMessage: (err) => {
            if (typeof err === 'string') return err;
            const error = err as z.infer<typeof UnifiedRouterErrorSchema>;
            return error.message || error.error || JSON.stringify(err);
          },
        }),
        successfulResponseHandler: createJsonResponseHandler(
          UnifiedRouterImageGenerationResponseSchema,
        ),
        abortSignal: options.abortSignal,
        fetch: this.config.fetch,
      });

      const res = UnifiedRouterImageGenerationResponseSchema.parse(response);
      const imageData = res.data[0];
      if (!imageData) throw new Error('No image data in response');

      // Convert base64 to Uint8Array if available
      let imageDataArray: Uint8Array | undefined;
      if (imageData.b64_json) {
        const binaryString = atob(imageData.b64_json);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        imageDataArray = bytes;
      }

      return {
        text: undefined,
        finishReason: 'stop' as LanguageModelV1FinishReason,
        usage: {
          promptTokens: res.usage?.prompt_tokens ?? 0,
          completionTokens: res.usage?.completion_tokens ?? 0,
        },
        rawCall: { rawPrompt: options.prompt, rawSettings: args },
        rawResponse: { headers: responseHeaders },
        warnings: [],
        files: imageDataArray ? [{
          data: imageDataArray,
          mimeType: 'image/png'
        }] : undefined,
        providerMetadata: res.provider_metadata as LanguageModelV1ProviderMetadata,
      };
    }

    // Handle video generation mode
    if (options.mode?.type === 'object-json' && this.modelId.includes('sora')) {
      const { responseHeaders, value: response } = await postJsonToApi({
        url: this.config.url({ path: '/videos/generations', modelId: this.modelId }),
        headers: this.config.headers(),
        body: args,
        failedResponseHandler: createJsonErrorResponseHandler({
          errorSchema: UnifiedRouterErrorSchema,
          errorToMessage: (err) => {
            if (typeof err === 'string') return err;
            const error = err as z.infer<typeof UnifiedRouterErrorSchema>;
            return error.message || error.error || JSON.stringify(err);
          },
        }),
        successfulResponseHandler: createJsonResponseHandler(
          UnifiedRouterVideoGenerationResponseSchema,
        ),
        abortSignal: options.abortSignal,
        fetch: this.config.fetch,
      });

      const res = UnifiedRouterVideoGenerationResponseSchema.parse(response);
      const videoData = res.data[0];
      if (!videoData) throw new Error('No video data in response');

      // Convert base64 to Uint8Array if available
      let videoDataArray: Uint8Array | undefined;
      if (videoData.b64_json) {
        const binaryString = atob(videoData.b64_json);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        videoDataArray = bytes;
      }

      return {
        text: undefined,
        finishReason: 'stop' as LanguageModelV1FinishReason,
        usage: {
          promptTokens: res.usage?.prompt_tokens ?? 0,
          completionTokens: res.usage?.completion_tokens ?? 0,
        },
        rawCall: { rawPrompt: options.prompt, rawSettings: args },
        rawResponse: { headers: responseHeaders },
        warnings: [],
        files: videoDataArray ? [{
          data: videoDataArray,
          mimeType: 'video/mp4'
        }] : undefined,
        providerMetadata: res.provider_metadata as LanguageModelV1ProviderMetadata,
      };
    }

    // Handle regular chat completion mode
    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({ path: '/chat/completions', modelId: this.modelId }),
      headers: this.config.headers(),
      body: args,
      failedResponseHandler: createJsonErrorResponseHandler({
        errorSchema: UnifiedRouterErrorSchema,
        errorToMessage: (err) => {
          if (typeof err === 'string') return err;
          const error = err as z.infer<typeof UnifiedRouterErrorSchema>;
          return error.message || error.error || JSON.stringify(err);
        },
      }),
      successfulResponseHandler: createJsonResponseHandler(
        UnifiedRouterChatCompletionResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });
    const res = UnifiedRouterChatCompletionResponseSchema.parse(response) as ExtendedResponse;
    const choice = res.choices[0];
    if (!choice) throw new Error('No choice in response');
    
    const message = choice.message;
    return {
      text: message.content ?? undefined,
      finishReason: (choice.finish_reason ?? 'other') as LanguageModelV1FinishReason,
      usage: {
        promptTokens: res.usage?.prompt_tokens ?? 0,
        completionTokens: res.usage?.completion_tokens ?? 0,
      },
      rawCall: { rawPrompt: options.prompt, rawSettings: args },
      rawResponse: { headers: responseHeaders },
      warnings: [],
      reasoning: message.reasoning,
      logprobs: message.logprobs as LanguageModelV1LogProbs,
      files: undefined,
      toolCalls: message.tool_calls?.map(tool => ({
        toolCallType: 'function',
        toolCallId: tool.id,
        toolName: tool.function.name,
        args: tool.function.arguments
      })) as LanguageModelV1FunctionToolCall[],
      providerMetadata: res.provider_metadata as LanguageModelV1ProviderMetadata,
    };
  }

  async doStream(options: LanguageModelV1CallOptions): Promise<{
    stream: ReadableStream<LanguageModelV1StreamPart>;
    rawCall: { rawPrompt: unknown; rawSettings: Record<string, unknown> };
    rawResponse?: { headers?: Record<string, string> };
    request?: { body?: string };
    warnings?: import('@ai-sdk/provider').LanguageModelV1CallWarning[];
  }> {
    const args = {
      model: this.modelId,
      stream: true,
      ...this.settings.extraBody,
      ...options,
    };

    const headers = this.config.headers();
    const filteredHeaders: Record<string, string> = {};
    for (const [key, value] of Object.entries(headers)) {
      if (value !== undefined) {
        filteredHeaders[key] = value;
      }
    }

    const response = await fetch(this.config.url({ path: '/chat/completions', modelId: this.modelId }), {
      method: 'POST',
      headers: filteredHeaders,
      body: JSON.stringify(args),
      signal: options.abortSignal,
    });

    if (!response.ok) {
      const error = await response.json();
      const parsedError = UnifiedRouterErrorSchema.parse(error);
      throw new Error(parsedError.message || parsedError.error || JSON.stringify(error));
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    const transformStream = new TransformStream({
      transform: async (chunk: Uint8Array, controller) => {
        const text = new TextDecoder().decode(chunk);
        const lines = text.split('\n').filter(line => line.trim() !== '');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            
            try {
              const parsedChunk = UnifiedRouterStreamChunkSchema.parse(JSON.parse(data));
              for (const choice of parsedChunk.choices) {
                if (choice.delta?.content) {
                  controller.enqueue({ type: 'text-delta', textDelta: choice.delta.content });
                }
                if (choice.delta?.tool_calls) {
                  for (const tool of choice.delta.tool_calls) {
                    controller.enqueue({ 
                      type: 'tool-call-delta',
                      toolCallType: 'function',
                      toolCallId: tool.id,
                      toolName: tool.function.name,
                      argsTextDelta: tool.function.arguments
                    });
                  }
                }
                if (choice.delta?.reasoning) {
                  controller.enqueue({ type: 'reasoning', textDelta: choice.delta.reasoning });
                }
                if (choice.delta?.logprobs) {
                  controller.enqueue({ 
                    type: 'response-metadata',
                    modelId: this.modelId,
                    timestamp: new Date()
                  });
                }
                if (choice.finish_reason) {
                  controller.enqueue({
                    type: 'finish',
                    finishReason: (choice.finish_reason ?? 'other') as LanguageModelV1FinishReason,
                    usage: { promptTokens: 0, completionTokens: 0 },
                    providerMetadata: parsedChunk.provider_metadata as LanguageModelV1ProviderMetadata
                  });
                }
              }
            } catch (error) {
              console.error('Error parsing chunk:', error);
            }
          }
        }
      }
    });

    return {
      stream: response.body.pipeThrough(transformStream) as ReadableStream<LanguageModelV1StreamPart>,
      rawCall: { rawPrompt: options.prompt, rawSettings: args },
      rawResponse: { headers: Object.fromEntries(response.headers.entries()) },
      request: { body: JSON.stringify(args) },
      warnings: [],
    };
  }
} 