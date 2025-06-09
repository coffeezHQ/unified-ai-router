import { LanguageModelV1 } from '@ai-sdk/provider';
import { UnifiedRouterLanguageModel } from './unified-router-language-model';

export interface UnifiedRouterProviderConfig {
  apiKey: string;
  model?: string;
  baseURL?: string;
}

export function createUnifiedRouter(config: UnifiedRouterProviderConfig): {
  provider: string;
  createModel: (modelId: string) => LanguageModelV1;
} {
  return {
    provider: 'unified-ai-router',
    createModel: (modelId: string): LanguageModelV1 => {
      return new UnifiedRouterLanguageModel({
        provider: 'unified-ai-router',
        model: modelId,
        apiKey: config.apiKey,
        baseURL: config.baseURL,
      });
    },
  };
}

export const unifiedrouter = createUnifiedRouter({
  apiKey: process.env.UNIFIEDROUTER_API_KEY || '',
}); 