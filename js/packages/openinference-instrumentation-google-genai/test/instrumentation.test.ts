import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { GoogleGenAIInstrumentation, isPatched } from "../src/instrumentation";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

describe("GoogleGenAIInstrumentation", () => {
  let instrumentation: GoogleGenAIInstrumentation;
  let exporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
    provider.register();

    instrumentation = new GoogleGenAIInstrumentation({
      tracerProvider: provider,
    });
  });

  afterEach(async () => {
    instrumentation.disable();
    exporter.reset();
    await provider.shutdown();
  });

  describe("initialization", () => {
    it("should create instrumentation instance", () => {
      expect(instrumentation).toBeDefined();
      expect(instrumentation.instrumentationName).toBe(
        "@arizeai/openinference-instrumentation-google-genai"
      );
    });

    it("should track patched state", () => {
      // Initially not patched (module not loaded)
      expect(isPatched()).toBe(false);
    });
  });

  describe("configuration", () => {
    it("should accept custom tracer provider", () => {
      const customProvider = new NodeTracerProvider();
      const inst = new GoogleGenAIInstrumentation({
        tracerProvider: customProvider,
      });
      expect(inst).toBeDefined();
    });

    it("should accept trace config", () => {
      const inst = new GoogleGenAIInstrumentation({
        traceConfig: {},
      });
      expect(inst).toBeDefined();
    });
  });
});
