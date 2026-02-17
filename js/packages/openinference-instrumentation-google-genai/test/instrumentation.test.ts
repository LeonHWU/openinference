import { setPromptTemplate, setSession } from "@arizeai/openinference-core";

import { context } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { GoogleGenAIInstrumentation, isPatched } from "../src";

import { vi, describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";

// Create a mock response that will be returned by the mock HTTP layer
let mockGenerateContentResponse: unknown = null;
let mockGenerateContentStreamResponse: unknown = null;
let mockError: Error | null = null;

// Mock Google GenAI module structure with a mock HTTP layer
class MockModels {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async generateContent(_params: any): Promise<any> {
    if (mockError) {
      throw mockError;
    }
    return mockGenerateContentResponse;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async generateContentStream(_params: any): Promise<any> {
    if (mockError) {
      throw mockError;
    }
    return mockGenerateContentStreamResponse;
  }
}

class MockGoogleGenAI {
  models: MockModels;
  constructor(_config?: { apiKey?: string }) {
    this.models = new MockModels();
  }
}

// Create a mock module that matches the expected structure
const mockGoogleGenAIModule = {
  GoogleGenAI: MockGoogleGenAI,
  Models: MockModels,
};

const memoryExporter = new InMemorySpanExporter();

describe("GoogleGenAIInstrumentation", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new GoogleGenAIInstrumentation();
  instrumentation.disable();
  let genai: MockGoogleGenAI;

  instrumentation.setTracerProvider(tracerProvider);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = mockGoogleGenAIModule;

  beforeAll(() => {
    instrumentation.enable();
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
    mockError = null;
    mockGenerateContentResponse = null;
    mockGenerateContentStreamResponse = null;
    genai = new MockGoogleGenAI({
      apiKey: "fake-api-key",
    });
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("is patched", () => {
    expect(isPatched()).toBe(true);
  });

  it("sets a patched flag correctly to track whether or not google-genai is instrumented", () => {
    instrumentation.disable();
    expect(isPatched()).toBe(false);
    instrumentation.enable();
    expect(isPatched()).toBe(true);
  });

  it("creates a span for generateContent", async () => {
    mockGenerateContentResponse = {
      responseId: "resp-123",
      modelVersion: "gemini-1.5-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ text: "Hello! How can I help you today?" }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 8,
        totalTokenCount: 18,
      },
    };

    await genai.models.generateContent({
      model: "gemini-1.5-flash",
      contents: [
        { role: "user", parts: [{ text: "Say hello" }] },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("GenerateContent");
    expect(span.attributes["openinference.span.kind"]).toBe("LLM");
    expect(span.attributes["llm.system"]).toBe("google_genai");
    expect(span.attributes["llm.provider"]).toBe("google");
    expect(span.attributes["llm.model_name"]).toBe("gemini-1.5-flash");
    expect(span.attributes["llm.token_count.prompt"]).toBe(10);
    expect(span.attributes["llm.token_count.completion"]).toBe(8);
    expect(span.attributes["llm.token_count.total"]).toBe(18);
    expect(span.attributes["llm.input_messages.0.message.role"]).toBe("user");
    expect(span.attributes["llm.input_messages.0.message.content"]).toBe("Say hello");
    expect(span.attributes["llm.output_messages.0.message.role"]).toBe("model");
    expect(span.attributes["llm.output_messages.0.message.content"]).toBe("Hello! How can I help you today?");
  });

  it("creates a span for generateContentStream", async () => {
    // Create a mock async iterable stream
    const chunks = [
      {
        candidates: [
          {
            content: {
              role: "model",
              parts: [{ text: "Hello" }],
            },
          },
        ],
      },
      {
        candidates: [
          {
            content: {
              role: "model",
              parts: [{ text: " World!" }],
            },
          },
        ],
      },
      {
        candidates: [
          {
            content: {
              role: "model",
              parts: [{ text: "" }],
            },
          },
        ],
        usageMetadata: {
          promptTokenCount: 5,
          candidatesTokenCount: 3,
          totalTokenCount: 8,
        },
        responseId: "stream-resp-123",
      },
    ];

    let index = 0;
    mockGenerateContentStreamResponse = {
      [Symbol.asyncIterator]() {
        return {
          async next() {
            if (index < chunks.length) {
              const value = chunks[index++];
              return { value, done: false };
            }
            return { value: undefined, done: true };
          },
        };
      },
    };

    const stream = await genai.models.generateContentStream({
      model: "gemini-1.5-flash",
      contents: [
        { role: "user", parts: [{ text: "Say hello world" }] },
      ],
    });

    let fullText = "";
    for await (const chunk of stream) {
      if (chunk.candidates?.[0]?.content?.parts) {
        for (const part of chunk.candidates[0].content.parts) {
          if (part.text) {
            fullText += part.text;
          }
        }
      }
    }

    expect(fullText).toBe("Hello World!");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("GenerateContentStream");
    expect(span.attributes["openinference.span.kind"]).toBe("LLM");
    expect(span.attributes["llm.system"]).toBe("google_genai");
    expect(span.attributes["llm.provider"]).toBe("google");
    expect(span.attributes["output.value"]).toBe("Hello World!");
    expect(span.attributes["llm.output_messages.0.message.content"]).toBe("Hello World!");
    expect(span.attributes["llm.token_count.prompt"]).toBe(5);
    expect(span.attributes["llm.token_count.completion"]).toBe(3);
    expect(span.attributes["llm.token_count.total"]).toBe(8);
  });

  it("captures tool/function calls in response", async () => {
    mockGenerateContentResponse = {
      responseId: "resp-tool-123",
      modelVersion: "gemini-1.5-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [
              {
                functionCall: {
                  name: "get_weather",
                  args: { location: "Boston" },
                },
              },
            ],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 20,
        candidatesTokenCount: 15,
        totalTokenCount: 35,
      },
    };

    await genai.models.generateContent({
      model: "gemini-1.5-flash",
      contents: [
        { role: "user", parts: [{ text: "What's the weather in Boston?" }] },
      ],
      tools: [
        {
          functionDeclarations: [
            {
              name: "get_weather",
              description: "Get the weather for a location",
              parameters: {
                type: "object",
                properties: {
                  location: { type: "string" },
                },
              },
            },
          ],
        },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"]).toBe("get_weather");
    expect(span.attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]).toBe('{"location":"Boston"}');
  });

  it("should not emit a span if tracing is suppressed", async () => {
    mockGenerateContentResponse = {
      responseId: "resp-suppressed",
      modelVersion: "gemini-1.5-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ text: "This should not be traced." }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 5,
        candidatesTokenCount: 6,
        totalTokenCount: 11,
      },
    };

    await new Promise((resolve) => {
      context.with(suppressTracing(context.active()), () => {
        resolve(
          genai.models.generateContent({
            model: "gemini-1.5-flash",
            contents: [
              { role: "user", parts: [{ text: "Test suppressed tracing" }] },
            ],
          }),
        );
      });
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(0);
  });

  it("should capture context attributes and add them to spans", async () => {
    mockGenerateContentResponse = {
      responseId: "resp-context",
      modelVersion: "gemini-1.5-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ text: "Context test response" }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 5,
        totalTokenCount: 15,
      },
    };

    await context.with(
      setSession(
        setPromptTemplate(context.active(), {
          template: "hello {name}",
          variables: { name: "world" },
          version: "V1.0",
        }),
        { sessionId: "session-id-123" },
      ),
      async () => {
        await genai.models.generateContent({
          model: "gemini-1.5-flash",
          contents: [
            { role: "user", parts: [{ text: "Test context propagation" }] },
          ],
        });
      },
    );

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("GenerateContent");
    expect(span.attributes["llm.prompt_template.template"]).toBe("hello {name}");
    expect(span.attributes["llm.prompt_template.variables"]).toBe('{"name":"world"}');
    expect(span.attributes["llm.prompt_template.version"]).toBe("V1.0");
    expect(span.attributes["session.id"]).toBe("session-id-123");
  });

  it("should capture generationConfig in invocation parameters", async () => {
    mockGenerateContentResponse = {
      responseId: "resp-config",
      modelVersion: "gemini-1.5-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ text: "Configured response" }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 5,
        totalTokenCount: 15,
      },
    };

    await genai.models.generateContent({
      model: "gemini-1.5-flash",
      contents: [
        { role: "user", parts: [{ text: "Test with config" }] },
      ],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 1000,
        topP: 0.9,
      },
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.attributes["llm.invocation_parameters"]).toBe(
      '{"temperature":0.7,"maxOutputTokens":1000,"topP":0.9}'
    );
  });

  it("should handle errors and record exception", async () => {
    mockError = new Error("API rate limit exceeded");

    await expect(
      genai.models.generateContent({
        model: "gemini-1.5-flash",
        contents: [
          { role: "user", parts: [{ text: "This will fail" }] },
        ],
      }),
    ).rejects.toThrow("API rate limit exceeded");

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.status.code).toBe(2); // SpanStatusCode.ERROR
    expect(span.status.message).toBe("API rate limit exceeded");
    expect(span.events.length).toBe(1);
    expect(span.events[0].name).toBe("exception");
  });
});

describe("GoogleGenAIInstrumentation with TraceConfig", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new GoogleGenAIInstrumentation({
    traceConfig: { hideInputs: true },
  });
  instrumentation.disable();
  let genai: MockGoogleGenAI;

  instrumentation.setTracerProvider(tracerProvider);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = mockGoogleGenAIModule;

  beforeAll(() => {
    instrumentation.enable();
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
    mockError = null;
    mockGenerateContentResponse = null;
    genai = new MockGoogleGenAI({
      apiKey: "fake-api-key",
    });
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should respect hideInputs trace config and mask input attributes", async () => {
    mockGenerateContentResponse = {
      responseId: "resp-hidden",
      modelVersion: "gemini-1.5-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ text: "This output should be visible" }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 6,
        totalTokenCount: 16,
      },
    };

    await genai.models.generateContent({
      model: "gemini-1.5-flash",
      contents: [
        { role: "user", parts: [{ text: "This input should be hidden" }] },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("GenerateContent");
    // Input should be redacted
    expect(span.attributes["input.value"]).toBe("__REDACTED__");
    // Output should still be visible
    expect(span.attributes["llm.output_messages.0.message.content"]).toBe("This output should be visible");
  });
});

describe("GoogleGenAIInstrumentation with hideOutputs TraceConfig", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new GoogleGenAIInstrumentation({
    traceConfig: { hideOutputs: true },
  });
  instrumentation.disable();
  let genai: MockGoogleGenAI;

  instrumentation.setTracerProvider(tracerProvider);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = mockGoogleGenAIModule;

  beforeAll(() => {
    instrumentation.enable();
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
    mockError = null;
    mockGenerateContentResponse = null;
    genai = new MockGoogleGenAI({
      apiKey: "fake-api-key",
    });
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should respect hideOutputs trace config and mask output attributes", async () => {
    mockGenerateContentResponse = {
      responseId: "resp-hidden-output",
      modelVersion: "gemini-1.5-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ text: "This output should be hidden" }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 6,
        totalTokenCount: 16,
      },
    };

    await genai.models.generateContent({
      model: "gemini-1.5-flash",
      contents: [
        { role: "user", parts: [{ text: "This input should be visible" }] },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("GenerateContent");
    // Input should still be visible
    expect(span.attributes["llm.input_messages.0.message.content"]).toBe("This input should be visible");
    // Output should be redacted
    expect(span.attributes["output.value"]).toBe("__REDACTED__");
  });
});

describe("GoogleGenAIInstrumentation with custom TracerProvider", () => {
  const customMemoryExporter = new InMemorySpanExporter();
  const customTracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(customMemoryExporter)],
  });
  let genai: MockGoogleGenAI;

  // Instantiate instrumentation with the custom provider
  const instrumentation = new GoogleGenAIInstrumentation({
    tracerProvider: customTracerProvider,
  });
  instrumentation.disable();

  // Mock the module exports
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = mockGoogleGenAIModule;

  beforeAll(() => {
    instrumentation.enable();
  });

  afterAll(() => {
    instrumentation.disable();
  });

  beforeEach(() => {
    memoryExporter.reset();
    customMemoryExporter.reset();
    mockError = null;
    mockGenerateContentResponse = null;
    genai = new MockGoogleGenAI({
      apiKey: "fake-api-key",
    });
  });

  afterEach(() => {
    vi.resetAllMocks();
    vi.clearAllMocks();
  });

  it("should use the provided tracer provider instead of the global one", async () => {
    mockGenerateContentResponse = {
      responseId: "resp-custom-provider",
      modelVersion: "gemini-1.5-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ text: "Custom provider response" }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 5,
        totalTokenCount: 15,
      },
    };

    await genai.models.generateContent({
      model: "gemini-1.5-flash",
      contents: [
        { role: "user", parts: [{ text: "Test custom provider" }] },
      ],
    });

    const customSpans = customMemoryExporter.getFinishedSpans();
    const globalSpans = memoryExporter.getFinishedSpans();
    expect(customSpans.length).toBe(1);
    expect(globalSpans.length).toBe(0);
    const span = customSpans[0];
    expect(span.name).toBe("GenerateContent");
    expect(span.attributes["llm.provider"]).toBe("google");
    expect(span.attributes["llm.model_name"]).toBe("gemini-1.5-flash");
  });
});

describe("GoogleGenAIInstrumentation initialization", () => {
  it("should create instrumentation instance", () => {
    const instrumentation = new GoogleGenAIInstrumentation();
    expect(instrumentation).toBeDefined();
    expect(instrumentation.instrumentationName).toBe(
      "@arizeai/openinference-instrumentation-google-genai"
    );
    instrumentation.disable();
  });

  it("should accept custom tracer provider in constructor", () => {
    const customProvider = new NodeTracerProvider();
    const inst = new GoogleGenAIInstrumentation({
      tracerProvider: customProvider,
    });
    expect(inst).toBeDefined();
    inst.disable();
  });

  it("should accept trace config in constructor", () => {
    const inst = new GoogleGenAIInstrumentation({
      traceConfig: { hideInputs: true, hideOutputs: true },
    });
    expect(inst).toBeDefined();
    inst.disable();
  });

  it("should track patched state correctly", () => {
    const inst = new GoogleGenAIInstrumentation();
    // Initially not patched (module not loaded)
    expect(isPatched()).toBe(false);
    inst.disable();
  });
});
