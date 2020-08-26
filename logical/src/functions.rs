fn nand(first: bool, second: bool) -> bool {
  !(first && second)
}

pub fn or(first: bool, second: bool) -> bool {
  nand(nand(first, first), nand(second, second))
}

#[cfg(test)]
mod tests {
  use super::*;  

  #[test]
  fn it_executes_or_operation() {
    assert_eq!(or(true, true), true);
    assert_eq!(or(true, false), true);
    assert_eq!(or(false, true), true);
    assert_eq!(or(false, false), false);
  }   
}
