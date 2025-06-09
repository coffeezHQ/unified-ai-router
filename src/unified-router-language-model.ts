import {
  LanguageModelV1,
  LanguageModelV1CallOptions,
  LanguageModelV1FinishReason,
  LanguageModelV1Message,
  LanguageModelV1ProviderMetadata,
  LanguageModelV1StreamPart,
  LanguageModelV1TextPart,
  LanguageModelV1ImagePart,
  LanguageModelV1FilePart,
} from '@ai-sdk/provider';

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

interface UnifiedRouterStreamResponse {
  choices: Array<{
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
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
  };
  provider_metadata?: LanguageModelV1ProviderMetadata;
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
    options: LanguageModelV1CallOptions,
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
    const response = await fetch(this.baseURL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
        ...this.headers,
      },
      body: JSON.stringify({
        model: this.model,
        messages: options.prompt,
        ...options,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error?.message || 'API request failed');
    }

    const data = await response.json();

    // Check if this is an image/video generation response
    if (data.data?.[0]?.b64_json) {
      const isVideo = this.model.toLowerCase().includes('video');
      return {
        text: '',
        finishReason: 'stop',
        usage: {
          promptTokens: data.usage?.prompt_tokens ?? 10,
          completionTokens: data.usage?.completion_tokens ?? 0,
          totalTokens: data.usage?.total_tokens ?? 10,
        },
        rawCall: {
          rawPrompt: options.prompt,
          rawSettings: { model: this.model },
        },
        rawResponse: {
          headers: Object.fromEntries(response.headers.entries()),
        },
        files: [{
          data: Buffer.from(data.data[0].b64_json, 'base64'),
          mimeType: isVideo ? 'video/mp4' : 'image/png',
        }],
        providerMetadata: data.provider_metadata as LanguageModelV1ProviderMetadata,
      };
    }

    // Validate response structure for text generation
    if (!Array.isArray(data.choices) || data.choices.length === 0) {
      throw new Error('Invalid response format: missing or empty choices array');
    }

    const choice = data.choices[0];
    if (!choice || typeof choice !== 'object') {
      throw new Error('Invalid response format: invalid choice object');
    }

    // Validate message content
    const content = choice.message?.content;
    if (content !== undefined && typeof content !== 'string') {
      throw new Error('Invalid response format: message content must be a string');
    }

    // Validate finish reason
    const finishReason = choice.finish_reason;
    if (finishReason !== undefined && typeof finishReason !== 'string') {
      throw new Error('Invalid response format: finish_reason must be a string');
    }

    return {
      text: content,
      finishReason: finishReason as LanguageModelV1FinishReason,
      usage: {
        promptTokens: data.usage?.prompt_tokens ?? 0,
        completionTokens: data.usage?.completion_tokens ?? 0,
        totalTokens: data.usage?.total_tokens ?? 0,
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
              const parsed = JSON.parse(data) as UnifiedRouterStreamResponse;
              if (parsed.choices?.[0]?.delta?.content) {
                controller.enqueue({
                  type: 'text-delta',
                  textDelta: parsed.choices[0].delta.content,
                });
              } else if (parsed.choices?.[0]?.delta?.tool_calls?.[0]) {
                const toolCall = parsed.choices[0].delta.tool_calls[0];
                controller.enqueue({
                  type: 'tool-call-delta',
                  toolCallType: 'function' as const,
                  toolCallId: toolCall.id,
                  toolName: toolCall.function.name,
                  argsTextDelta: toolCall.function.arguments,
                });
              } else if (parsed.choices?.[0]?.finish_reason) {
                const finishChunk: {
                  type: 'finish';
                  finishReason: LanguageModelV1FinishReason;
                  usage: {
                    promptTokens: number;
                    completionTokens: number;
                  };
                  providerMetadata?: LanguageModelV1ProviderMetadata;
                } = {
                  type: 'finish',
                  finishReason: parsed.choices[0].finish_reason as LanguageModelV1FinishReason,
                  usage: {
                    promptTokens: parsed.usage?.prompt_tokens ?? 0,
                    completionTokens: parsed.usage?.completion_tokens ?? 0,
                  },
                };

                if (parsed.provider_metadata) {
                  finishChunk.providerMetadata = parsed.provider_metadata;
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