export type UnifiedRouterModelId = string;

export interface UnifiedRouterSettings {
  apiKey: string;
  baseURL?: string;
  extraBody?: Record<string, unknown>;
  headers?: Record<string, string>;
}

export interface UnifiedRouterProviderConfig {
  provider: string;
  compatibility: 'strict' | 'compatible';
  headers: () => Record<string, string | undefined>;
  url: (options: { modelId: string; path: string }) => string;
  fetch?: typeof fetch;
  extraBody?: Record<string, unknown>;
} 