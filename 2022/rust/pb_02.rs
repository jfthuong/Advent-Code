use std::fs;

const INPUT_PATH: &str = "../inputs/02.txt";

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Gesture {
    Rock = 0,
    Paper = 1,
    Scissor = 2,
}

fn get_points(other_letter: &str, my_letter: &str) -> i32 {
    // TODO: Maybe find a better way to match letters to Gesture
    // ... and specify type to be only one of 3 letters
    let other = match other_letter {
        "A" => Gesture::Rock,
        "B" => Gesture::Paper,
        "C" => Gesture::Scissor,
        &_ => panic!("Invalid letter for other"),
    };
    let me = match my_letter {
        "X" => Gesture::Rock,
        "Y" => Gesture::Paper,
        "Z" => Gesture::Scissor,
        &_ => panic!("Invalid letter for myself"),
    };

    let points_choice = (me.clone() as i32) + 1;
    if other == me {
        return 3 + points_choice;
    }

    if (me as i32) == ((other as i32) + 1) % 3 {
        return 6 + points_choice;
    }

    return points_choice;
}

fn main() {
    let point = get_points("A", "Y");
    println!("-- Point: {point}");

    let contents = fs::read_to_string(INPUT_PATH).expect("Should be readable input file");
    let points = contents
        .lines()
        .map(|line| {
            let mut parts = line.split_whitespace();
            let other_letter = parts.next().unwrap();
            let my_letter = parts.next().unwrap();
            get_points(other_letter, my_letter)
        })
        .sum::<i32>();
    println!("Solution of part1: {}", points);
}