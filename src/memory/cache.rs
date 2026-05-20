use regex::{Regex, RegexSet};
use std::collections::HashMap;

use super::allocator::allocate_init;

fn is_synthetic_runtime_weight(name: &str) -> bool {
    matches!(
        name,
        "model.position_embedding.weight" | "model.rotary_emb.weight"
    )
}

#[derive(Debug, Clone)]
pub struct Cache<T> {
    pub storge: HashMap<String, *mut T>,
    regex_set: RegexSet,
    layer_regex: Regex,
    parameters: HashMap<String, Vec<T>>,
    parameter_loading_enabled: bool,
}

impl<T> Cache<T>
where
    T: Copy + Default
    // + Send
    // + Sync

{
    pub fn new(parameters: HashMap<String, Vec<T>>) -> Self {
        let regex_set = RegexSet::new(&[
            r".*weight",
            r"model\.layers\.\d+\.self_attn\.(key_position_tensor|value_tensor)",
            r"model\.layers\.\d+\.(.*)",
            r"model.*.output",
        ])
        .unwrap();
        let layer_regex = Regex::new(r"model\.layers\.\d+\.(.*)").unwrap();

        let parameter_loading_enabled = !parameters.is_empty();

        Self {
            storge: HashMap::new(),
            regex_set: regex_set,
            layer_regex: layer_regex,
            parameters: parameters,
            parameter_loading_enabled,
        }
    }

    pub fn get(&mut self, name: &str, size: usize) -> *mut T {
        // Generate keyname from name
        // 1. For parameter end with weight
        // 2. For k, v
        // 3. For intermediate variables of different layers, they share the same keyname.
        // 4. other output
        // For example, "model.layers.0.abc.def" and "model.layers.1.abc.def" are two intermediate
        // variables of layers. They share the same keyname "abc.def"

        for m in self.regex_set.matches(name).iter() {
            println!("name {} ", name);
            let p = match m {
                0 => {
                    // parameters
                    // println!("parameters {} ", name);

                    match self.parameters.remove(name) {
                        Some(data) => {
                            assert_eq!(
                                data.len(),
                                size,
                                "Parameter {} shape mismatch: safetensors has {} elements, tensor expects {}",
                                name,
                                data.len(),
                                size
                            );
                            // 将Vec<T>转换为Box<[T]>然后泄露到堆上获取指针
                            let boxed_slice = data.into_boxed_slice();
                            let data_ptr: *mut T = Box::leak(boxed_slice).as_mut_ptr();
                            self.storge.insert(name.to_owned(), data_ptr);
                            data_ptr
                        }
                        None if is_synthetic_runtime_weight(name) => {
                            allocate_init(size, T::default())
                        }
                        None if !self.parameter_loading_enabled => {
                            allocate_init(size, T::default())
                        }
                        None => panic!("Parameter {} not found in loaded safetensors", name),
                    }
                }
                1 => {
                    // kv
                    // println!("kv {}", name);
                    let data_ptr: *mut T = allocate_init(size, T::default());
                    self.storge.insert(name.to_owned(), data_ptr);
                    data_ptr
                }
                2 => match self.layer_regex.captures(name) {
                    // layer
                    Some(capture) => {
                        let key_name = capture.get(1).unwrap().as_str();
                        // println!("layer {} {}", name, key_name);
                        match self.storge.get(key_name) {
                            Some(ptr) => ptr.clone(),
                            None => {
                                let data_ptr: *mut T = allocate_init(size, T::default());
                                self.storge.insert(key_name.to_owned(), data_ptr);
                                data_ptr
                            }
                        }
                    }
                    None => panic!(),
                },
                3 => {
                    // other
                    // println!("kv {}", name);
                    let data_ptr: *mut T = allocate_init(size, T::default());
                    self.storge.insert(name.to_owned(), data_ptr);
                    data_ptr
                }

                _ => panic!(),
            };
            return p;
        }
        panic!()
    }


}
/* 
unsafe impl <T:Copy + Default
    + Send
    + Sync> Send for Cache<T> {}
unsafe impl <T:Copy + Default
    + Send
    + Sync> Sync for Cache<T> {}
    */
// pub static mut cache: Lazy<Cache> = Lazy::new(|| {Cache::new()});

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "model.layers.0.self_attn.v_proj.weight".to_string(),
            vec![1.0f32, 2.0f32],
        );
        parameters.insert("model.embd.weight".to_string(), vec![3.0f32, 4.0f32]);

        let mut cache: Cache<f32> = Cache::new(parameters);

        let name1 = String::from("model.layers.0.self_attn.v_proj.weight");
        let name2 = String::from("model.embd.weight");
        let p1 = cache.get(&name1, 2);
        let p2 = cache.get(&name2, 2);
        // println!("{:?}", p1);
        assert_ne!(p1, p2);

        let k = String::from("model.layers.0.self_attn.key_position_tensor");
        let v = String::from("model.layers.0.self_attn.mapv_tensor");
        let p1 = cache.get(&k, 2);
        let p2 = cache.get(&v, 2);
        // println!("{:?}", p1);
        assert_ne!(p1, p2);

        let layer1 = String::from("model.layers.0.self_attn.cc");
        let layer2 = String::from("model.layers.1.self_attn.cc");
        let p1 = cache.get(&layer1, 2);
        let p2 = cache.get(&layer2, 2);
        // println!("{:?}", p1);
        assert_eq!(p1, p2);
    }
}
