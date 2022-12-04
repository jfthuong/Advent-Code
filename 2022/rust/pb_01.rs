use std::fs;

const INPUT_PATH: &str = "../inputs/01.txt";
const EOL_DOUBLE: &str = "\r\n\r\n";  // We are using Windows line endings

fn main() {
    let contents = fs::read_to_string(INPUT_PATH).expect("Should be readable input file");
    let mut calories: Vec<i32> = contents
        .split(EOL_DOUBLE)
        .map(|block| {
            block
                .trim()
                .lines()
                .map(|line| line.parse::<i32>().expect("Should be a number"))
                .sum()
        })
        .collect();

        println!("Calories #0: {:?}", calories[0]);

    calories.sort();
    calories.reverse();

    // Maximum value
    let max = calories[0];
    println!("-- Max: {max}");

    // Sum of top 3 values
    let sum = calories[0] + calories[1] + calories[2];
    println!("-- Sum of top 3: {sum}");

    // We check that results are the same as in Python
    assert_eq!(max, 69693);
    assert_eq!(sum, 200945);

}
