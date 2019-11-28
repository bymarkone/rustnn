use std::io::{Read};
use std::fs::File;
use byteorder::{ReadBytesExt, BigEndian};

pub fn load() -> Vec<u8> {
  return load_labels();
}

fn load_labels() -> Vec<u8> {
  let mut f = File::open("./data/train-labels-idx1-ubyte").unwrap();
  
  let magic_nr = f.read_u32::<BigEndian>().unwrap();
  assert!(magic_nr == 0x0801);
  let nelems = f.read_u32::<BigEndian>().unwrap() as usize;
  assert!(nelems > 0);

  let mut labels = Vec::with_capacity(nelems);

  for byte in f.bytes().take(nelems) {
      let lbl = byte.unwrap();
      assert!(lbl <= 9);
      labels.push(lbl);
  }

  println!("Found {} labels", labels.len());
  assert!(labels.len() == nelems);

  return labels;
}

fn load_images() {

}
