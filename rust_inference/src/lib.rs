use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

#[derive(Deserialize)]
struct ModelData {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f32>,
    classes: Vec<String>,
    weights: Vec<Vec<f32>>,
    intercepts: Vec<f32>,
}

#[pyclass]
pub struct RustClassifier {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f32>,
    classes: Vec<String>,
    weights: Vec<Vec<f32>>,
    intercepts: Vec<f32>,
}

fn extract_char_wb_ngrams(text: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let mut ngrams = Vec::new();
    for word in text.split_whitespace() {
        let padded = format!(" {} ", word);
        let chars: Vec<char> = padded.chars().collect();
        let len = chars.len();
        for n in min_n..=max_n {
            if len < n {
                continue;
            }
            for i in 0..=(len - n) {
                let ngram: String = chars[i..i + n].iter().collect();
                ngrams.push(ngram);
            }
        }
    }
    ngrams
}

#[pymethods]
impl RustClassifier {
    #[new]
    pub fn new(weights_path: String) -> PyResult<Self> {
        let mut file = File::open(weights_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let data: ModelData = serde_json::from_str(&contents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(RustClassifier {
            vocabulary: data.vocabulary,
            idf: data.idf,
            classes: data.classes,
            weights: data.weights,
            intercepts: data.intercepts,
        })
    }

    pub fn predict(&self, text: String) -> String {
        let ngrams = extract_char_wb_ngrams(&text, 3, 5);

        let mut term_counts: HashMap<usize, f32> = HashMap::new();
        for ngram in ngrams {
            if let Some(&index) = self.vocabulary.get(&ngram) {
                *term_counts.entry(index).or_insert(0.0) += 1.0;
            }
        }

        let mut tfidf_vector: Vec<(usize, f32)> = Vec::with_capacity(term_counts.len());
        let mut l2_norm_sq: f32 = 0.0;
        for (index, tf) in term_counts {
            let idf = self.idf[index];
            let val = tf * idf;
            tfidf_vector.push((index, val));
            l2_norm_sq += val * val;
        }

        let l2_norm = l2_norm_sq.sqrt();
        if l2_norm > 0.0 {
            for (_, val) in tfidf_vector.iter_mut() {
                *val /= l2_norm;
            }
        }

        let num_classes = self.classes.len();
        let mut best_class_index = 0;
        let mut best_score = f32::NEG_INFINITY;

        for c in 0..num_classes {
            let mut score = self.intercepts[c];
            let row = &self.weights[c];
            for &(feat_idx, val) in &tfidf_vector {
                score += val * row[feat_idx];
            }
            if score > best_score {
                best_score = score;
                best_class_index = c;
            }
        }

        self.classes[best_class_index].clone()
    }
}

#[pymodule]
fn rust_inference(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustClassifier>()?;
    Ok(())
}
