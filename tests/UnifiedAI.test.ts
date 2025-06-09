import { UnifiedAI } from '../src/UnifiedAI';
import { createUnifiedRouter } from '../src/unifiedRouterProvider';

describe('UnifiedAI', () => {
  const mockProvider = createUnifiedRouter({ apiKey: 'test-key' });
  const anotherProvider = createUnifiedRouter({ apiKey: 'another-key' });

  it('should initialize with default provider', () => {
    const ai = new UnifiedAI({
      providers: { mockProvider, anotherProvider },
      defaultProvider: 'mockProvider',
    });
    expect(ai.model('test-model')).toMatchObject({
      provider: 'unified-ai-router',
      model: 'test-model',
    });
  });

  it('should switch providers', () => {
    const ai = new UnifiedAI({
      providers: { mockProvider, anotherProvider },
      defaultProvider: 'mockProvider',
    });
    ai.useProvider('anotherProvider');
    expect(ai.model('test-model').config.apiKey).toBe('another-key');
  });

  it('should throw if provider not found', () => {
    const ai = new UnifiedAI({
      providers: { mockProvider },
      defaultProvider: 'mockProvider',
    });
    expect(() => ai.useProvider('missing')).toThrow();
  });

  it('should call generateText with correct model', async () => {
    const ai = new UnifiedAI({
      providers: { mockProvider },
      defaultProvider: 'mockProvider',
    });
    const fakeGenerateText = jest.fn(async ({ model, prompt }) => ({ text: `Echo: ${prompt}` }));
    const result = await ai.generateText(fakeGenerateText, { model: 'test-model', prompt: 'hi' });
    expect(result.text).toBe('Echo: hi');
    expect(fakeGenerateText).toHaveBeenCalledWith({ model: expect.any(Object), prompt: 'hi' });
  });

  it('should call streamText with correct model', async () => {
    const ai = new UnifiedAI({
      providers: { mockProvider },
      defaultProvider: 'mockProvider',
    });
    const fakeStreamText = jest.fn(async ({ model, messages }) => ({ stream: true, messages }));
    const result = await ai.streamText(fakeStreamText, { model: 'test-model', messages: [{ role: 'user', content: 'hi' }] });
    expect(result.stream).toBe(true);
    expect(fakeStreamText).toHaveBeenCalledWith({ model: expect.any(Object), messages: [{ role: 'user', content: 'hi' }] });
  });

  it('should fallback to default provider if provider is not specified', () => {
    const ai = new UnifiedAI({
      providers: { mockProvider, anotherProvider },
      defaultProvider: 'mockProvider',
    });
    expect(ai.getProvider()).toBe(mockProvider);
  });

  it('should throw if no providers configured', () => {
    expect(() => new UnifiedAI({ providers: {}, defaultProvider: 'none' })).not.toThrow();
    const ai = new UnifiedAI({ providers: {}, defaultProvider: 'none' });
    expect(() => ai.getProvider()).toThrow();
  });

  it('should switch back and forth between providers', () => {
    const ai = new UnifiedAI({
      providers: { mockProvider, anotherProvider },
      defaultProvider: 'mockProvider',
    });
    ai.useProvider('anotherProvider');
    expect(ai.model('test-model').config.apiKey).toBe('another-key');
    ai.useProvider('mockProvider');
    expect(ai.model('test-model').config.apiKey).toBe('test-key');
  });

  it('should pass modelOptions through model()', () => {
    const ai = new UnifiedAI({
      providers: { mockProvider },
      defaultProvider: 'mockProvider',
    });
    const model = ai.model('test-model', { extraBody: { foo: 'bar' }, custom: 42 });
    expect(model.modelOptions).toEqual({ extraBody: { foo: 'bar' }, custom: 42 });
  });
}); 