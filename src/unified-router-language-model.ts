import { LanguageModelV1, LanguageModelV1CallOptions, LanguageModelV1StreamPart, LanguageModelV1FinishReason, LanguageModelV1ProviderMetadata } from '@ai-sdk/provider';

interface UnifiedRouterResponse {
  id: string;
  model: string;
  created: number;
  object: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
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

export class UnifiedRouterLanguageModel implements LanguageModelV1 {
  public readonly specificationVersion = 'v1';
  public readonly modelId: string;
  public readonly defaultObjectGenerationMode = 'tool';
  public provider: string;
  private model: string;
  private apiKey: string;
  private baseURL: string;

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
    this.baseURL = config.baseURL || 'https://api.unified-ai-router.com/v1';
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
  }> {
    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages: [{ role: 'user', content: options.prompt }],
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        stream: false,
      }),
    });

    const data = await response.json() as UnifiedRouterResponse | UnifiedRouterError;

    if ('error' in data) {
      throw new Error(`Unified Router API error: ${data.error.message}`);
    }

    const choice = data.choices[0];
    return {
      text: choice.message.content,
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
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages: [{ role: 'user', content: options.prompt }],
        temperature: options.temperature,
        max_tokens: options.maxTokens,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json() as UnifiedRouterError;
      throw new Error(`Unified Router API error: ${errorData.error.message}`);
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
            if (data === '[DONE]') {
              controller.close();
              return;
            }

            try {
              const parsed = JSON.parse(data) as UnifiedRouterResponse;
              const content = parsed.choices[0].message.content;

              if (content) {
                controller.enqueue({ type: 'text-delta', textDelta: content });
              }
            } catch (error) {
              console.error('Error parsing stream chunk:', error);
            }
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
      request: {
        body: JSON.stringify({
          model: this.model,
          messages: [{ role: 'user', content: options.prompt }],
          temperature: options.temperature,
          max_tokens: options.maxTokens,
          stream: true,
        }),
      },
      warnings: [],
    };
  }
} 