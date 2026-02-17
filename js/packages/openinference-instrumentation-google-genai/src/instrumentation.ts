import {
  OITracer,
  safelyJSONStringify,
  TraceConfigOptions,
} from "@arizeai/openinference-core";
import {
  LLMProvider,
  LLMSystem,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import {
  Attributes,
  context,
  diag,
  Span,
  SpanKind,
  SpanStatusCode,
  trace,
  Tracer,
  TracerProvider,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  safeExecuteInTheMiddle,
} from "@opentelemetry/instrumentation";

// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - No version file until build
import { VERSION } from "./version";

import type { GoogleGenAI } from "@google/genai";

const MODULE_NAME = "@google/genai";
const INSTRUMENTATION_NAME = "@arizeai/openinference-instrumentation-google-genai";

/**
 * Flag to check if the google-genai module has been patched
 */
let _isOpenInferencePatched = false;

/**
 * function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isOpenInferencePatched;
}

/**
 * Resolves the execution context for the current span
 */
function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing
    ? trace.setSpan(context.active(), span)
    : activeContext;
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}

interface GoogleGenAIModuleExports {
  GoogleGenAI: typeof GoogleGenAI;
}

/**
 * An auto instrumentation class for Google GenAI that creates OpenInference compliant spans
 */
export class GoogleGenAIInstrumentation extends InstrumentationBase<GoogleGenAIModuleExports> {
  private oiTracer: OITracer;
  private tracerProvider?: TracerProvider;
  private traceConfig?: TraceConfigOptions;

  constructor({
    instrumentationConfig,
    traceConfig,
    tracerProvider,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
    tracerProvider?: TracerProvider;
  } = {}) {
    super(
      INSTRUMENTATION_NAME,
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.oiTracer = new OITracer({
      tracer:
        this.tracerProvider?.getTracer(INSTRUMENTATION_NAME, VERSION) ??
        this.tracer,
      traceConfig,
    });
  }

  protected init(): InstrumentationModuleDefinition<GoogleGenAIModuleExports> {
    const module = new InstrumentationNodeModuleDefinition<GoogleGenAIModuleExports>(
      MODULE_NAME,
      [">=1.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the Google GenAI module
   */
  manuallyInstrument(module: GoogleGenAIModuleExports) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  get tracer(): Tracer {
    if (this.tracerProvider) {
      return this.tracerProvider.getTracer(
        this.instrumentationName,
        this.instrumentationVersion,
      );
    }
    return super.tracer;
  }

  setTracerProvider(tracerProvider: TracerProvider): void {
    super.setTracerProvider(tracerProvider);
    this.tracerProvider = tracerProvider;
    this.oiTracer = new OITracer({
      tracer: this.tracer,
      traceConfig: this.traceConfig,
    });
  }

  /**
   * Patches the Google GenAI module
   */
  private patch(
    module: GoogleGenAIModuleExports & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    if (module?.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }

    // Handle ES module default export structure
    const genaiModule =
      (module as GoogleGenAIModuleExports & { default?: GoogleGenAIModuleExports }).default ||
      module;

    // Patch models.generateContent
    this.patchGenerateContent(genaiModule);

    // Patch models.generateContentStream
    this.patchGenerateContentStream(genaiModule);

    _isOpenInferencePatched = true;
    try {
      module.openInferencePatched = true;
    } catch (e) {
      diag.debug(`Failed to set ${MODULE_NAME} patched flag on the module`, e);
    }

    return module;
  }

  private patchGenerateContent(module: GoogleGenAIModuleExports) {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation = this;

    try {
      // The GoogleGenAI client structure: client.models.generateContent()
      // We need to patch the Models class prototype
      const ModelsClass = this.getModelsClass(module);
      if (!ModelsClass) {
        diag.warn("Could not find Models class to patch generateContent");
        return;
      }

      this._wrap(
        ModelsClass.prototype,
        "generateContent",
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (original: any): any => {
          return function patchedGenerateContent(
            this: unknown,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ...args: any[]
          ) {
            const requestParams = args[0] || {};
            const span = instrumentation.oiTracer.startSpan("GenerateContent", {
              kind: SpanKind.INTERNAL,
              attributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
                [SemanticConventions.LLM_SYSTEM]: "google_genai",
                [SemanticConventions.LLM_PROVIDER]: "google",
                [SemanticConventions.INPUT_VALUE]: safelyJSONStringify(requestParams),
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                ...getRequestAttributes(requestParams),
              },
            });

            const execContext = getExecContext(span);
            const execPromise = safeExecuteInTheMiddle(
              () => {
                return context.with(trace.setSpan(execContext, span), () => {
                  return original.apply(this, args);
                });
              },
              (error: Error | undefined) => {
                if (error) {
                  span.recordException(error);
                  span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: error.message,
                  });
                  span.end();
                }
              },
            );

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const wrappedPromiseThen = (result: any) => {
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]: safelyJSONStringify(result),
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                ...getResponseAttributes(result),
              });
              span.setStatus({ code: SpanStatusCode.OK });
              span.end();
              return result;
            };

            const wrappedPromise = execPromise
              .then(wrappedPromiseThen)
              .catch((error: Error) => {
                span.recordException(error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
                throw error;
              });

            return context.bind(execContext, wrappedPromise);
          };
        },
      );

      diag.debug("Patched models.generateContent");
    } catch (e) {
      diag.warn("Failed to patch generateContent", e);
    }
  }

  private patchGenerateContentStream(module: GoogleGenAIModuleExports) {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation = this;

    try {
      const ModelsClass = this.getModelsClass(module);
      if (!ModelsClass) {
        diag.warn("Could not find Models class to patch generateContentStream");
        return;
      }

      this._wrap(
        ModelsClass.prototype,
        "generateContentStream",
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (original: any): any => {
          return function patchedGenerateContentStream(
            this: unknown,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ...args: any[]
          ) {
            const requestParams = args[0] || {};
            const span = instrumentation.oiTracer.startSpan("GenerateContentStream", {
              kind: SpanKind.INTERNAL,
              attributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
                [SemanticConventions.LLM_SYSTEM]: "google_genai",
                [SemanticConventions.LLM_PROVIDER]: "google",
                [SemanticConventions.INPUT_VALUE]: safelyJSONStringify(requestParams),
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                ...getRequestAttributes(requestParams),
              },
            });

            const execContext = getExecContext(span);
            const execPromise = safeExecuteInTheMiddle(
              () => {
                return context.with(trace.setSpan(execContext, span), () => {
                  return original.apply(this, args);
                });
              },
              (error: Error | undefined) => {
                if (error) {
                  span.recordException(error);
                  span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: error.message,
                  });
                  span.end();
                }
              },
            );

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const wrappedPromiseThen = (stream: any) => {
              // Wrap the stream to capture output
              return wrapStream(stream, span);
            };

            const wrappedPromise = execPromise
              .then(wrappedPromiseThen)
              .catch((error: Error) => {
                span.recordException(error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
                throw error;
              });

            return context.bind(execContext, wrappedPromise);
          };
        },
      );

      diag.debug("Patched models.generateContentStream");
    } catch (e) {
      diag.warn("Failed to patch generateContentStream", e);
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private getModelsClass(module: GoogleGenAIModuleExports): any {
    try {
      // Try to get the Models class from the module exports
      // The structure might vary based on how the module is bundled
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const anyModule = module as any;

      // Try different paths to find the Models class
      if (anyModule.Models) {
        return anyModule.Models;
      }

      // For the GoogleGenAI client, models is a property that creates Models instances
      // We might need to patch at the prototype level of the client
      if (anyModule.GoogleGenAI?.prototype) {
        // Return a proxy that allows us to access the models property
        return this.getModelsFromClient(anyModule.GoogleGenAI);
      }

      return null;
    } catch (e) {
      diag.debug("Error getting Models class", e);
      return null;
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private getModelsFromClient(GoogleGenAIClass: any): any {
    // Create a temporary instance to get the models property type
    // This is a workaround for module structures where Models isn't directly exported
    try {
      const descriptor = Object.getOwnPropertyDescriptor(
        GoogleGenAIClass.prototype,
        "models"
      );
      if (descriptor?.get) {
        // It's a getter, we need to patch differently
        return null;
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  /**
   * Un-patches the Google GenAI module
   */
  private unpatch(
    moduleExports: GoogleGenAIModuleExports & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    try {
      const ModelsClass = this.getModelsClass(moduleExports);
      if (ModelsClass) {
        this._unwrap(ModelsClass.prototype, "generateContent");
        this._unwrap(ModelsClass.prototype, "generateContentStream");
      }
    } catch (e) {
      diag.warn("Failed to unpatch", e);
    }

    _isOpenInferencePatched = false;
    try {
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}

/**
 * Extract attributes from the request
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getRequestAttributes(params: any): Attributes {
  const attributes: Attributes = {};

  if (params.model) {
    attributes[SemanticConventions.LLM_MODEL_NAME] = params.model;
  }

  if (params.contents) {
    // Extract input messages
    const contents = Array.isArray(params.contents) ? params.contents : [params.contents];
    contents.forEach((content: any, index: number) => {
      const prefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.`;
      if (content.role) {
        attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = content.role;
      }
      if (content.parts) {
        const textParts = content.parts
          .filter((p: any) => p.text)
          .map((p: any) => p.text)
          .join("");
        if (textParts) {
          attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] = textParts;
        }
      }
    });
  }

  if (params.generationConfig) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] = safelyJSONStringify(
      params.generationConfig
    );
  }

  if (params.tools) {
    params.tools.forEach((tool: any, index: number) => {
      const toolJson = safelyJSONStringify(tool);
      if (toolJson) {
        attributes[
          `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`
        ] = toolJson;
      }
    });
  }

  return attributes;
}

/**
 * Extract attributes from the response
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getResponseAttributes(response: any): Attributes {
  const attributes: Attributes = {};

  if (response.modelVersion) {
    attributes[SemanticConventions.LLM_MODEL_NAME] = response.modelVersion;
  }

  if (response.responseId) {
    attributes["gen_ai.response.id"] = response.responseId;
  }

  // Extract usage metadata
  if (response.usageMetadata) {
    const usage = response.usageMetadata;
    if (usage.promptTokenCount !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = usage.promptTokenCount;
    }
    if (usage.candidatesTokenCount !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = usage.candidatesTokenCount;
    }
    if (usage.totalTokenCount !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = usage.totalTokenCount;
    }
  }

  // Extract output messages
  if (response.candidates && response.candidates.length > 0) {
    const candidate = response.candidates[0];
    if (candidate.content) {
      const prefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;
      attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] =
        candidate.content.role || "model";

      if (candidate.content.parts) {
        const textParts = candidate.content.parts
          .filter((p: any) => p.text)
          .map((p: any) => p.text)
          .join("");
        if (textParts) {
          attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] = textParts;
        }

        // Handle function calls
        candidate.content.parts.forEach((part: any, partIndex: number) => {
          if (part.functionCall) {
            const toolCallPrefix = `${prefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${partIndex}.`;
            attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] =
              part.functionCall.name;
            attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] =
              safelyJSONStringify(part.functionCall.args);
          }
        });
      }
    }
  }

  return attributes;
}

/**
 * Wrap a streaming response to capture chunks
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function wrapStream(stream: any, span: Span): any {
  let fullText = "";
  let usageMetadata: any = null;
  let responseId: string | undefined;

  const originalIterator = stream[Symbol.asyncIterator].bind(stream);

  stream[Symbol.asyncIterator] = function () {
    const iterator = originalIterator();

    return {
      async next() {
        const result = await iterator.next();

        if (!result.done && result.value) {
          const chunk = result.value;

          // Accumulate text
          if (chunk.candidates?.[0]?.content?.parts) {
            for (const part of chunk.candidates[0].content.parts) {
              if (part.text) {
                fullText += part.text;
              }
            }
          }

          // Capture usage metadata from final chunk
          if (chunk.usageMetadata) {
            usageMetadata = chunk.usageMetadata;
          }

          // Capture response ID
          if (chunk.responseId) {
            responseId = chunk.responseId;
          }
        }

        if (result.done) {
          // Stream finished, set attributes and end span
          const attributes: Attributes = {
            [SemanticConventions.OUTPUT_VALUE]: fullText,
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
              "model",
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
              fullText,
          };

          if (responseId) {
            attributes["gen_ai.response.id"] = responseId;
          }

          if (usageMetadata) {
            if (usageMetadata.promptTokenCount !== undefined) {
              attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] =
                usageMetadata.promptTokenCount;
            }
            if (usageMetadata.candidatesTokenCount !== undefined) {
              attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] =
                usageMetadata.candidatesTokenCount;
            }
            if (usageMetadata.totalTokenCount !== undefined) {
              attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] =
                usageMetadata.totalTokenCount;
            }
          }

          span.setAttributes(attributes);
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
        }

        return result;
      },
    };
  };

  return stream;
}
