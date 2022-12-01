use std::fs;

const INPUT_PATH: &str = "../inputs/01_calories.txt";

fn main() {
    let contents = fs::read_to_string(INPUT_PATH).expect("Cannot read input file");
    let mut calories: Vec<i32> = contents
        // let calories: Vec<Vec<&str>> = contents
        .split("\r\n\r\n")
        .map(|block| {
            block
                .trim()
                .split("\r\n")
                .map(|line| line.parse::<i32>().expect("Should be a number"))
                .sum()
        })
        .collect();

    println!("Calories #0: {:?}", calories[0]);

    // Maximum value
    let max = calories.iter().max().expect("Should have a max value");
    println!("Max: {}", max);

    // Sum of top 3 values
    calories.sort();
    calories.reverse();
    let sum = calories[0] + calories[1] + calories[2];
    println!("Sum of top 3: {}", sum);
}
