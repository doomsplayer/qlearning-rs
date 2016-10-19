#![feature(link_args)]

#[link_args = "-L/usr/local/lib/gcc/6"]
extern "C" {}

#[macro_use]
extern crate rulinalg;
extern crate rand;
extern crate itertools;

use itertools::{Itertools, MinMaxResult};

use rand::distributions::{IndependentSample, Range, Weighted, WeightedChoice};
use rand::ThreadRng;

use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;



/**
reward: matrix of     a  c  t  i  o  n  
                   s  0 -1  0  1  2  3
                   t -1 10  2  1  0 -1
                   a  0 -1  -1 0  1  2
                   t  0  0  1  2  3 -1
                   e  0  0  0  1 10 -1
                   
where action is actually the next state, which is to say, 
we currently only implemented an action only corresponds to absolute 1 single next state

**/
pub fn optimize(reward: Matrix<f64>,
                gamma: f64,
                initial_state: &[usize],
                target_state: &[usize],
                time: Option<usize>)
                -> Matrix<f64> {
    assert!(0f64 <= gamma && gamma <= 1f64);

    let time = time.unwrap_or(100);

    let nactions = reward.cols();
    let nstates = reward.rows();
    let mut q = Matrix::zeros(nstates, nactions);


    let mut rng = rand::thread_rng();
    for _ in 0..time {
        let mut weighted_initial_state: Vec<_> = initial_state.iter()
            .map(|&s| {
                Weighted {
                    weight: 1,
                    item: s,
                }
            })
            .collect();
        let wc = WeightedChoice::new(&mut weighted_initial_state);
        let mut current_state = wc.ind_sample(&mut rng);

        // run a round first to avoid the situation that we already get to target state
        current_state = update_q(&mut q, &reward, current_state, gamma);

        while let None = target_state.iter().find(|&&t| t == current_state) {
            current_state = update_q(&mut q, &reward, current_state, gamma);
        }
    }
    q
}

fn update_q(q: &mut Matrix<f64>, reward: &Matrix<f64>, current_state: usize, gamma: f64) -> usize {

    let t = available_actions(&reward, current_state);

    let (next_state, immediate_reward) = choose_random_action(&t);
    let t = available_actions(&reward, next_state);

    q[[current_state, next_state]] = match t.into_iter()
        .minmax_by(|&(state, _), &(state2, _)| {
            q[[next_state, state]].partial_cmp(&q[[next_state, state2]]).unwrap()
        }) {
        MinMaxResult::OneElement((state, _)) => {
            gamma * q[[next_state, state]] as f64 + immediate_reward as f64
        }
        MinMaxResult::MinMax(_, (state, _)) => {
            gamma * q[[next_state, state]] as f64 + immediate_reward as f64
        }
        MinMaxResult::NoElements => unreachable!("shit!"),
    };
    next_state
}

fn available_actions(reward: &Matrix<f64>, state: usize) -> Vec<(usize, f64)> {
    reward.get_row(state)
        .unwrap()
        .iter()
        .enumerate()
        .filter(|&(_, &reward)| reward != -1f64)
        .map(|(a, &b)| (a, b))
        .collect()
}

fn choose_random_action(actions: &[(usize, f64)]) -> (usize, f64) {
    let mut rng: ThreadRng = rand::thread_rng();
    let between = Range::new(0, actions.len());
    let action_idx: usize = between.ind_sample(&mut rng);
    actions[action_idx]
}

#[cfg(test)]
mod tests {
    use rulinalg::matrix::{BaseMatrix, Matrix};
    #[test]
    fn test_room_search() {
        let reward = matrix! {
            -1.0, -1.0, -1.0, -1.0,  0.0,  -1.0;
            -1.0, -1.0, -1.0,  0.0, -1.0, 100.0;
            -1.0, -1.0, -1.0,  0.0, -1.0,  -1.0;
            -1.0,  0.0,  0.0, -1.0,  0.0,  -1.0;
             0.0, -1.0, -1.0,  0.0, -1.0, 100.0;
            -1.0,  0.0, -1.0, -1.0,  0.0, 100.0
        };
        let q = ::optimize(reward.into(), 0.8f64, &[0, 1, 2, 3, 4, 5], &[5], Some(1000));
        let q = Matrix::from_fn(q.rows(),
                                q.cols(),
                                |row, col| q[[row, col]].round() as usize);
        let true_result = matrix! {
            0, 0, 0, 0, 320, 0;
            0, 0, 0, 400, 0, 400;
            0, 0, 0, 256, 0, 0;
            0, 320, 320, 0, 320, 0;
            400, 0, 0, 400, 0, 400;
            0, 500, 0, 0, 500, 500
        };
        assert_eq!(true_result, q);
    }
}
