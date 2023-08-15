
pub fn main() -> anyhow::Result<()> {
    let tokenizer = std::fs::read("tokenizer.json")?;
    let model = std::fs::read("model.bin")?;
    let model_data = torch::llma2::worker::ModelData { tokenizer, model };
    let model = torch::llma2::worker::Model::load(model_data)?;
    model.run(0.0, "She likes to read Shakespear".to_string())?;

    Ok(())
}