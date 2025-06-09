import { LanguageModelV1, LanguageModelV1CallOptions, LanguageModelV1StreamPart, LanguageModelV1ImagePart, LanguageModelV1FinishReason, LanguageModelV1ProviderMetadata, LanguageModelV1ObjectGenerationMode } from '@ai-sdk/provider';
import { z } from 'zod';

const UnifiedRouterResponseSchema = z.object({
  id: z.string(),
  model: z.string(),
  choices: z.array(z.object({
    index: z.number(),
    message: z.object({
      role: z.string(),
      content: z.string(),
    }),
    finish_reason: z.string().nullable(),
  })),
  usage: z.object({
    prompt_tokens: z.number(),
    completion_tokens: z.number(),
    total_tokens: z.number(),
    cost: z.number().optional(),
  }).optional(),
  provider_metadata: z.record(z.unknown()).optional(),
});

const UnifiedRouterImageGenerationResponseSchema = z.object({
  id: z.string(),
  model: z.string(),
  data: z.array(z.object({
    url: z.string().optional(),
    b64_json: z.string().optional(),
    revised_prompt: z.string().optional(),
  })),
  usage: z.object({
    prompt_tokens: z.number(),
    completion_tokens: z.number(),
    total_tokens: z.number(),
    cost: z.number().optional(),
  }).optional(),
  provider_metadata: z.record(z.unknown()).optional(),
});

export class UnifiedRouterLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1';
  readonly modelId: string;
  readonly defaultObjectGenerationMode: LanguageModelV1ObjectGenerationMode = 'tool';
  readonly provider: string;

  private readonly apiKey: string;
  private readonly baseURL: string;
  private readonly extraBody: Record<string, unknown>;

  constructor({
    model,
    apiKey,
    baseURL = 'https://api.unifiedrouter.ai/v1',
    extraBody = {},
    provider = 'unified-ai-router',
  }: {
    model: string;
    apiKey: string;
    baseURL?: string;
    extraBody?: Record<string, unknown>;
    provider?: string;
  }) {
    this.modelId = model;
    this.apiKey = apiKey;
    this.baseURL = baseURL;
    this.extraBody = extraBody;
    this.provider = provider;
  }

  async doGenerate(
    options: LanguageModelV1CallOptions,
  ) {
    const { prompt, mode } = options;

    // Handle image generation
    if (mode?.type === 'object-json') {
      const response = await fetch(`${this.baseURL}/images/generations`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.modelId,
          prompt: prompt,
          n: 1,
          size: '1024x1024',
          ...this.extraBody,
        }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Image generation failed: ${error.error || response.statusText}`);
      }

      const data = await response.json();
      const validatedData = UnifiedRouterImageGenerationResponseSchema.parse(data);

      // Convert base64 to Uint8Array if available
      const imageData = validatedData.data[0].b64_json
        ? Buffer.from(validatedData.data[0].b64_json, 'base64')
        : null;

      return {
        text: '',
        finishReason: 'stop' as LanguageModelV1FinishReason,
        usage: {
          promptTokens: validatedData.usage?.prompt_tokens ?? 0,
          completionTokens: validatedData.usage?.completion_tokens ?? 0,
        },
        rawCall: { rawPrompt: prompt, rawSettings: { model: this.modelId, ...this.extraBody } },
        rawResponse: { headers: Object.fromEntries(response.headers.entries()) },
        warnings: [],
        files: imageData ? [{
          data: imageData,
          mimeType: 'image/png',
        }] : undefined,
        providerMetadata: validatedData.provider_metadata as LanguageModelV1ProviderMetadata,
      };
    }

    // Handle text generation
    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.modelId,
        messages: [{ role: 'user', content: prompt }],
        ...this.extraBody,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Text generation failed: ${error.error || response.statusText}`);
    }

    const data = await response.json();
    const validatedData = UnifiedRouterResponseSchema.parse(data);

    return {
      text: validatedData.choices[0].message.content,
      finishReason: (validatedData.choices[0].finish_reason ?? 'stop') as LanguageModelV1FinishReason,
      usage: {
        promptTokens: validatedData.usage?.prompt_tokens ?? 0,
        completionTokens: validatedData.usage?.completion_tokens ?? 0,
      },
      rawCall: { rawPrompt: prompt, rawSettings: { model: this.modelId, ...this.extraBody } },
      rawResponse: { headers: Object.fromEntries(response.headers.entries()) },
      warnings: [],
      providerMetadata: validatedData.provider_metadata as LanguageModelV1ProviderMetadata,
    };
  }

  async doStream(
    options: LanguageModelV1CallOptions,
  ) {
    const { prompt } = options;

    const response = await fetch(`${this.baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.modelId,
        messages: [{ role: 'user', content: prompt }],
        stream: true,
        ...this.extraBody,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Streaming failed: ${error.error || response.statusText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    const stream = new ReadableStream({
      async pull(controller) {
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
              const parsed = JSON.parse(data);
              const validatedData = UnifiedRouterResponseSchema.parse(parsed);
              const content = validatedData.choices[0].message.content;

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
      rawCall: { rawPrompt: prompt, rawSettings: { model: this.modelId, ...this.extraBody } },
      rawResponse: { headers: Object.fromEntries(response.headers.entries()) },
      request: { body: JSON.stringify({ model: this.modelId, messages: [{ role: 'user', content: prompt }], stream: true, ...this.extraBody }) },
      warnings: [],
    };
  }
} 