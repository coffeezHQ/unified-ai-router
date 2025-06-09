import { UnifiedRouterLanguageModel } from './unified-router-language-model';
import type { UnifiedRouterModelId, UnifiedRouterSettings, UnifiedRouterProviderConfig } from './types/unified-router-provider';

export function createUnifiedRouter(settings: UnifiedRouterSettings) {
  return (modelId: UnifiedRouterModelId) => {
    const config: UnifiedRouterProviderConfig = {
      provider: 'unified-ai-router',
      compatibility: 'compatible',
      headers: () => ({
        'Authorization': `Bearer ${settings.apiKey}`,
        'Content-Type': 'application/json',
        ...(settings.headers || {}),
      }),
      url: ({ modelId, path }) => settings.baseURL ? `${settings.baseURL}${path}` : `https://api.unifiedrouter.ai/v1${path}`,
      extraBody: settings.extraBody,
    };
    return new UnifiedRouterLanguageModel(modelId, settings, config);
  };
}

export const unifiedrouter = createUnifiedRouter({
  apiKey: process.env.UNIFIEDROUTER_API_KEY || '',
}); 