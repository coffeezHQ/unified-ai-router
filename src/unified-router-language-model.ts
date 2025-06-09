import { LanguageModelV1, LanguageModelV1CallOptions, LanguageModelV1StreamPart, LanguageModelV1FinishReason, LanguageModelV1ProviderMetadata, LanguageModelV1TextPart, LanguageModelV1Message } from '@ai-sdk/provider';

interface UnifiedRouterResponse {
  id: string;
  model: string;
  created: number;
  object: string;
  choices: Array<{
    index: number;
    message?: {
      role: string;
      content: string;
    };
    delta?: {
      content?: string;
      tool_calls?: Array<{
        id: string;
        type: string;
        function: {
          name: string;
          arguments: string;
        };
      }>;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  provider_metadata?: Record<string, unknown>;
}

interface UnifiedRouterError {
  error: {
    message: string;
    type: string;
    code: string;
  };
}

interface UnifiedRouterImageResponse {
  data: Array<{
    b64_json: string;
    url?: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  provider_metadata?: Record<string, unknown>;
}

export class UnifiedRouterLanguageModel implements LanguageModelV1 {
  public readonly specificationVersion = 'v1';
  public readonly modelId: string;
  public readonly defaultObjectGenerationMode = 'tool';
  public provider: string;
  private model: string;
  private apiKey: string;
  private baseURL: string;
  private headers: Record<string, string>;

  constructor(config: {
    provider: string;
    model: string;
    apiKey: string;
    baseURL?: string;
  }) {
    this.provider = config.provider;
    this.model = config.model;
    this.modelId = config.model;
    this.apiKey = config.apiKey;
    this.baseURL = config.baseURL || 'https://api.unifiedrouter.com/v1';
    this.headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`,
      'X-Test': 'test-value',
    };
  }

  async doGenerate(
    options: LanguageModelV1CallOptions
  ): Promise<{
    text?: string;
    finishReason: LanguageModelV1FinishReason;
    usage: {
      promptTokens: number;
      completionTokens: number;
      totalTokens: number;
    };
    rawCall: {
      rawPrompt: unknown;
      rawSettings: Record<string, unknown>;
    };
    rawResponse?: {
      headers: Record<string, string>;
    };
    warnings?: Array<{
      type: 'other';
      message: string;
    }>;
    providerMetadata?: LanguageModelV1ProviderMetadata;
    files?: Array<{
      data: Uint8Array;
      mimeType: string;
    }>;
  }> {
    // Check if this is an image/video generation request
    if (options.mode?.type === 'object-json') {
      const isVideo = this.model.toLowerCase().includes('video');
      const endpoint = isVideo ? '/videos/generations' : '/images/generations';
      const prompt = Array.isArray(options.prompt)
        ? (options.prompt[0]?.content && Array.isArray(options.prompt[0].content) && options.prompt[0].content[0]?.type === 'text'
            ? options.prompt[0].content[0].text
            : '')
        : '';
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          model: this.model,
          prompt,
        }),
      });
      const data = await response.json() as UnifiedRouterImageResponse;
      if (!data.data?.[0]) {
        throw new Error(isVideo ? 'Invalid video generation response' : 'Invalid image generation response');
      }
      const fileData = data.data[0];
      if (fileData.b64_json) {
        // base64 image/video
        const buffer = Uint8Array.from(Buffer.from(fileData.b64_json, 'base64'));
        return {
          files: [{ data: buffer, mimeType: isVideo ? 'video/mp4' : 'image/png' }],
          usage: {
            promptTokens: data.usage?.prompt_tokens ?? 0,
            completionTokens: data.usage?.completion_tokens ?? 0,
            totalTokens: data.usage?.total_tokens ?? 0,
          },
          finishReason: 'stop',
          rawCall: {
            rawPrompt: options.prompt,
            rawSettings: { model: this.model },
          },
          rawResponse: {
            headers: Object.fromEntries(response.headers.entries()),
          },
          warnings: undefined,
          providerMetadata: typeof data.provider_metadata === 'object' ? data.provider_metadata as LanguageModelV1ProviderMetadata : undefined,
        };
      } else {
        // Only support b64_json for now; if url, throw error or comment for future support
        throw new Error(isVideo ? 'Invalid video generation response' : 'Invalid image generation response');
      }
    }

    // Regular text generation
    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: this.model,
        messages: Array.isArray(options.prompt)
          ? (options.prompt as LanguageModelV1Message[]).map(msg => ({
              role: msg.role,
              content: (msg.content as LanguageModelV1TextPart[]).map(part => part.text).join('')
            }))
          : [{ role: 'user', content: options.prompt }],
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        stream: false,
      }),
    });

    const data = await response.json() as UnifiedRouterResponse | UnifiedRouterError;

    if ('error' in data) {
      throw new Error(`Unified Router API error: ${data.error?.message || 'Unknown error'}`);
    }

    const choice = data.choices[0];
    return {
      text: choice.message?.content,
      finishReason: choice.finish_reason as LanguageModelV1FinishReason,
      usage: {
        promptTokens: data.usage.prompt_tokens,
        completionTokens: data.usage.completion_tokens,
        totalTokens: data.usage.total_tokens,
      },
      rawCall: {
        rawPrompt: options.prompt,
        rawSettings: { model: this.model },
      },
      rawResponse: {
        headers: Object.fromEntries(response.headers.entries()),
      },
      warnings: [],
      providerMetadata: data.provider_metadata as LanguageModelV1ProviderMetadata,
    };
  }

  async doStream(
    options: LanguageModelV1CallOptions
  ): Promise<{
    stream: ReadableStream<LanguageModelV1StreamPart>;
    rawCall: {
      rawPrompt: unknown;
      rawSettings: Record<string, unknown>;
    };
    rawResponse?: {
      headers: Record<string, string>;
    };
    request?: {
      body: string;
    };
    warnings?: Array<{
      type: 'other';
      message: string;
    }>;
  }> {
    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: this.model,
        messages: Array.isArray(options.prompt)
          ? (options.prompt as LanguageModelV1Message[]).map(msg => ({
              role: msg.role,
              content: (msg.content as LanguageModelV1TextPart[]).map(part => part.text).join('')
            }))
          : [{ role: 'user', content: options.prompt }],
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json() as UnifiedRouterError;
      throw new Error(`Unified Router API error: ${errorData.error?.message || 'Unknown error'}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    const stream = new ReadableStream<LanguageModelV1StreamPart>({
      async pull(controller): Promise<void> {
        const { done, value } = await reader.read();

        if (done) {
          controller.close();
          return;
        }

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(Boolean);

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              if (parsed.choices?.[0]?.delta?.content) {
                controller.enqueue({
                  type: 'text-delta',
                  textDelta: parsed.choices[0].delta.content,
                });
              } else if (parsed.choices?.[0]?.delta?.tool_calls) {
                const toolCall = parsed.choices[0].delta.tool_calls[0];
                controller.enqueue({
                  type: 'tool-call-delta',
                  toolCallType: 'function' as const,
                  toolCallId: toolCall.id,
                  toolName: toolCall.function.name,
                  argsTextDelta: toolCall.function.arguments,
                });
              } else if (parsed.choices?.[0]?.finish_reason) {
                const finishChunk: any = {
                  type: 'finish',
                  finishReason: parsed.choices[0].finish_reason as LanguageModelV1FinishReason,
                };
                if (parsed.usage) {
                  finishChunk.usage = {
                    promptTokens: parsed.usage?.prompt_tokens ?? 0,
                    completionTokens: parsed.usage?.completion_tokens ?? 0,
                  };
                }
                if (typeof parsed.provider_metadata === 'object') {
                  finishChunk.providerMetadata = parsed.provider_metadata as LanguageModelV1ProviderMetadata;
                }
                controller.enqueue(finishChunk);
              }
            } catch (e) { /* ignore */ }
          }
        }
      },
    });

    return {
      stream,
      rawCall: {
        rawPrompt: options.prompt,
        rawSettings: { model: this.model },
      },
      rawResponse: {
        headers: Object.fromEntries(response.headers.entries()),
      },
    };
  }
} 