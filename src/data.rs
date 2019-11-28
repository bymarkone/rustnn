use ndarray::prelude::*;
use std::io::{Read, Result};
use std::fs::File;
use byteorder::{ReadBytesExt, BigEndian};

pub fn load() -> (Vec<u8>, Vec<Array2<u8>>) {
  return (load_labels(), load_images().unwrap());
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

  assert!(labels.len() == nelems);
  println!("Found {} labels", labels.len());

  return labels;
}

fn load_images() -> Result<Vec<Array2<u8>>> {
  let mut file = File::open("./data/train-images-idx3-ubyte")?;
  
  let magic_nr = file.read_u32::<BigEndian>()?;
  assert!(magic_nr == 0x0803);

  let nimages = file.read_u32::<BigEndian>()? as usize;
  assert!(nimages == 60000);
  
  let nrows = file.read_u32::<BigEndian>()? as usize;
  assert!(nrows == 28);

  let ncols = file.read_u32::<BigEndian>()? as usize;
  assert!(ncols == 28);

  let mut images = Vec::with_capacity(nimages);

  let mut buf = [0; 784];
  while let Ok(()) = file.read_exact(&mut buf) {
    let image = Array2::from_shape_vec((784, 1), buf.to_vec()).unwrap();
    images.push(image);
  }

  assert!(images.len() == nimages);
  println!("Found {} images", images.len());
  
  return Ok(images);
}
