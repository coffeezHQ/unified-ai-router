import axios from 'axios';
jest.mock('axios');

const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('unifiedRouterProvider', () => {
  afterEach(() => jest.clearAllMocks());

  it('should create a provider factory', () => {
    const { createUnifiedRouter } = require('../src/unifiedRouterProvider');
    const factory = createUnifiedRouter({ apiKey: 'abc', baseURL: 'https://test', extraBody: { foo: 'bar' } });
    const model = factory('openai/gpt-4o', { extraBody: { test: 1 } });
    expect(model).toMatchObject({
      provider: 'unified-router',
      model: 'openai/gpt-4o',
      config: { apiKey: 'abc', baseURL: 'https://test', extraBody: { foo: 'bar' } },
      modelOptions: { extraBody: { test: 1 } },
    });
  });

  it('should merge modelOptions correctly', () => {
    const { createUnifiedRouter } = require('../src/unifiedRouterProvider');
    const factory = createUnifiedRouter({ apiKey: 'abc' });
    const model = factory('openai/gpt-4o', { extraBody: { foo: 'bar' }, custom: 42 });
    expect(model.modelOptions).toEqual({ extraBody: { foo: 'bar' }, custom: 42 });
  });

  it('should allow baseURL override', () => {
    const { createUnifiedRouter } = require('../src/unifiedRouterProvider');
    const factory = createUnifiedRouter({ apiKey: 'abc', baseURL: 'https://custom' });
    const model = factory('openai/gpt-4o');
    expect(model.config.baseURL).toBe('https://custom');
  });

  it('should propagate extraBody from config and modelOptions', () => {
    const { createUnifiedRouter } = require('../src/unifiedRouterProvider');
    const factory = createUnifiedRouter({ apiKey: 'abc', extraBody: { a: 1 } });
    const model = factory('openai/gpt-4o', { extraBody: { b: 2 } });
    expect(model.config.extraBody).toEqual({ a: 1 });
    expect(model.modelOptions.extraBody).toEqual({ b: 2 });
  });

  it('should call baseURL with all data and return response', async () => {
    const { createUnifiedRouter } = require('../src/unifiedRouterProvider');
    const factory = createUnifiedRouter({ apiKey: 'abc', baseURL: 'https://test' });
    const model = factory('openai/gpt-4o', { extraBody: { foo: 'bar' } });
    mockedAxios.post.mockResolvedValueOnce({ data: { result: 'ok', echo: true } });
    const result = await model.callModel({ userInput: 'hi', chatContext: { id: 1 }, custom: 42 });
    expect(mockedAxios.post).toHaveBeenCalledWith(
      'https://test',
      expect.objectContaining({
        model: 'openai/gpt-4o',
        api_key: 'abc',
        user_input: 'hi',
        chat_context: { id: 1 },
        model_options: { extraBody: { foo: 'bar' } },
        config: expect.any(Object),
        custom: 42,
      }),
      expect.any(Object)
    );
    expect(result).toEqual({ result: 'ok', echo: true });
  });
}); 