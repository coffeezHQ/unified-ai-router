import { LanguageModelV1 } from '@ai-sdk/provider';
import { UnifiedRouterLanguageModel } from './unified-router-language-model';

export interface UnifiedRouterProviderConfig {
  apiKey: string;
  baseURL?: string;
  extraBody?: Record<string, unknown>;
}

export function createUnifiedRouter(config: UnifiedRouterProviderConfig) {
  return (model: string, modelOptions?: { extraBody?: Record<string, unknown> }) => {
    return new UnifiedRouterLanguageModel({
      model,
      apiKey: config.apiKey,
      baseURL: config.baseURL,
      extraBody: { ...config.extraBody, ...modelOptions?.extraBody },
      provider: 'unified-ai-router',
    });
  };
}

export const unifiedrouter = createUnifiedRouter({
  apiKey: process.env.UNIFIEDROUTER_API_KEY || '',
}); 