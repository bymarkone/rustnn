use ndarray::prelude::*;
use std::io::{Read, Result};
use std::fs::File;
use byteorder::{ReadBytesExt, BigEndian};

pub fn load() -> (Array2<u8>, Array1<u8>) {
  return (load_images().unwrap(), load_labels().unwrap());
}

fn load_labels() -> Result<Array1<u8>> {
  println!("Loading labels");
  let mut file = File::open("./data/train-labels-idx1-ubyte")?;
  
  let magic_nr = file.read_u32::<BigEndian>()?;
  assert!(magic_nr == 0x0801);
  let nelems = file.read_u32::<BigEndian>()? as usize;
  assert!(nelems > 0);

  let labels = Array::from_shape_vec(nelems, 
                  file.bytes().take(nelems).map(Result::unwrap).collect()).unwrap();

  assert!(labels.len() == nelems);
  println!("Found {} labels", labels.len());

  return Ok(labels);
}

fn load_images() -> Result<Array2<u8>> {
  println!("Loading images");
  let mut file = File::open("./data/train-images-idx3-ubyte")?;
  
  let magic_nr = file.read_u32::<BigEndian>()?;
  assert!(magic_nr == 0x0803);

  let nimages = file.read_u32::<BigEndian>()? as usize;
  assert!(nimages == 60000);
  
  let nrows = file.read_u32::<BigEndian>()? as usize;
  assert!(nrows == 28);

  let ncols = file.read_u32::<BigEndian>()? as usize;
  assert!(ncols == 28);

  let mut buf = Vec::new();
  file.read_to_end(&mut buf);
  
  let images = Array2::from_shape_vec((60000, 784), buf.to_vec()).unwrap();

  assert!(images.dim() == (nimages, nrows * ncols));
  println!("Found {:?} images", images.dim());
  
  return Ok(images);
}
