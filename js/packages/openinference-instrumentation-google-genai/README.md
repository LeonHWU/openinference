# OpenInference Instrumentation for Google GenAI

This module provides automatic instrumentation for the [Google GenAI SDK](https://www.npmjs.com/package/@google/genai) (`@google/genai`).

## Installation

```bash
npm install @arizeai/openinference-instrumentation-google-genai
```

## Usage

### Auto Instrumentation

```typescript
import { GoogleGenAIInstrumentation } from "@arizeai/openinference-instrumentation-google-genai";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

registerInstrumentations({
  instrumentations: [new GoogleGenAIInstrumentation()],
});
```

### Manual Instrumentation

For environments where auto-instrumentation may not work (e.g., ESM, bundlers):

```typescript
import { GoogleGenAIInstrumentation } from "@arizeai/openinference-instrumentation-google-genai";
import * as genai from "@google/genai";

const instrumentation = new GoogleGenAIInstrumentation();
instrumentation.manuallyInstrument(genai);
```

### With Custom Tracer Provider

```typescript
import { GoogleGenAIInstrumentation } from "@arizeai/openinference-instrumentation-google-genai";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

const provider = new NodeTracerProvider();
provider.register();

const instrumentation = new GoogleGenAIInstrumentation({
  tracerProvider: provider,
});
```

## Captured Spans

This instrumentation captures the following operations:

- `GenerateContent` - Non-streaming content generation
- `GenerateContentStream` - Streaming content generation

## Span Attributes

The following OpenInference semantic convention attributes are captured:

### Request Attributes

- `llm.model_name` - The model being used
- `llm.input_messages` - Input messages with role and content
- `llm.invocation_parameters` - Generation config parameters
- `llm.tools` - Tool definitions (if provided)
- `input.value` - Full request as JSON
- `input.mime_type` - "application/json"

### Response Attributes

- `llm.output_messages` - Output messages with role and content
- `llm.token_count.prompt` - Number of prompt tokens
- `llm.token_count.completion` - Number of completion tokens
- `llm.token_count.total` - Total token count
- `gen_ai.response.id` - Response ID
- `output.value` - Full response as JSON (non-streaming) or concatenated text (streaming)
- `output.mime_type` - MIME type of output

## Configuration

### Trace Config

You can configure trace behavior using `TraceConfigOptions`:

```typescript
const instrumentation = new GoogleGenAIInstrumentation({
  traceConfig: {
    // Add your trace configuration here
  },
});
```

## License

Apache-2.0
