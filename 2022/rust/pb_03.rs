use std::collections::HashSet;
use std::fs;

const INPUT_PATH: &str = "../inputs/03.txt";

fn letter_2_priority(letter: char) -> i32 {
    // Transform into priority a -> 1, b -> 2, ..., A -> 27, B -> 28, ...
    match letter {
        'a'..='z' => letter as i32 - 'a' as i32 + 1,
        'A'..='Z' => letter as i32 - 'A' as i32 + 27,
        _ => panic!("Invalid character"),
    }
}

fn get_line_priority(line: &str) -> i32 {
    // Split line in 2 equal parts as sets
    let (left, right) = line.split_at(line.len() / 2);
    // Get common letter between left and right using sets
    let left_set: HashSet<char> = left.chars().collect();
    let right_set: HashSet<char> = right.chars().collect();
    let common_characters: Vec<char> = left_set.intersection(&right_set).cloned().collect();
    letter_2_priority(common_characters[0])
}

fn get_priority_3_lines(line1: String, line2: String, line3: String) -> i32 {
    // Get intersection of 3 lines
    let sets: Vec<HashSet<char>> = [line1, line2, line3]
        .iter()
        .map(|line| line.chars().collect())
        .collect();

    let intersection = sets
        .iter()
        .skip(1)
        .fold(sets[0].clone(), |acc, hs| {
            acc.intersection(hs).cloned().collect()
        })
        .into_iter()
        .collect::<Vec<char>>();

    letter_2_priority(intersection[0])
}

fn main() {
    let contents = fs::read_to_string(INPUT_PATH).expect("Should be readable input file");
    let lines = contents.lines();

    let priority_sum = lines
        .clone()
        .map(|line| get_line_priority(line))
        .sum::<i32>();
    println!("Solution of part1: {}", priority_sum);

    // Read 3 lines at a time using chunks
    let iter_lines = lines.clone().chunks(3);
    let priority_sum = iter_lines
        .clone()
        .step_by(3)
        .zip(iter_lines.skip(1).step_by(3))
        .zip(iter_lines.skip(2).step_by(3))
        .map(|((line1, line2), line3)| {
            get_priority_3_lines(line1.to_string(), line2.to_string(), line3.to_string())
        })
        .sum::<i32>();
    println!("Solution of part2: {}", priority_sum);
}
