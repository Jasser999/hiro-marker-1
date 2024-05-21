importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

let model = null;

onmessage = async(event) => {
    const input = event.data;
    const output = await run_model(input);
    postMessage(output);
};

async function run_model(input) {
    if (!model) {
        model = await ort.InferenceSession.create("/static/scripts/UTS5000SAGEN_1_2_160px_fix.onnx");
    }
    input = new ort.Tensor(Float32Array.from(input), [1, 3, 160, 160]);
    const outputs = await model.run({ images: input });
    return outputs["output0"].data;
}
