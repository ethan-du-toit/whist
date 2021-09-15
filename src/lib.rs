#![allow(non_snake_case)]

use core::fmt;
use js_sys::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::cmp::max;
use std::convert::TryInto;
use std::iter;
use wasm_bindgen::prelude::*;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum Suit {
    Spades,
    Hearts,
    Diamonds,
    Clubs,
}

//const PLAYER_NAMES: [&str; 4] = ["South", "West", "North", "East"];
const TRUMP_SEQ: [Option<Suit>; 5] = [
    Some(Suit::Hearts),
    Some(Suit::Spades),
    Some(Suit::Diamonds),
    Some(Suit::Clubs),
    None,
];

const PLACEHOLDER: GameState = GameState {
    player: 0,
    handPlayer: 0,
    trump: Some(Suit::Hearts),
    hand: [
        Some(card(Suit::Hearts, 5)),
        Some(card(Suit::Hearts, 6)),
        Some(card(Suit::Hearts, 12)),
        Some(card(Suit::Spades, 1)),
        Some(card(Suit::Spades, 5)),
        Some(card(Suit::Spades, 8)),
        Some(card(Suit::Spades, 11)),
        Some(card(Suit::Diamonds, 1)),
        Some(card(Suit::Diamonds, 2)),
        Some(card(Suit::Diamonds, 7)),
        Some(card(Suit::Clubs, 0)),
        Some(card(Suit::Clubs, 7)),
        Some(card(Suit::Clubs, 11)),
    ],
    ruffs: 0,
    played: 0,
    trick: [None, None, None],
    tricksWonBy0: 0,
    sizes: [13, 13, 13, 13],
};

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct Card {
    suit: Suit,
    rank: u8, // 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A <- 2 is 0, ... A is 12
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rank: String = if self.rank == 12 {
            "A".to_string()
        } else if self.rank == 11 {
            "K".to_string()
        } else if self.rank == 10 {
            "Q".to_string()
        } else if self.rank == 9 {
            "J".to_string()
        } else {
            (self.rank + 2).to_string()
        };
        write!(f, "{} {:?}", rank, self.suit)
    }
}

#[derive(Debug, Clone)]
struct GameState {
    player: u8,
    handPlayer: u8,
    trump: Option<Suit>,
    hand: [Option<Card>; 13], // hand of handPlayer
    ruffs: u16,
    played: u64,
    trick: [Option<Card>; 3],
    tricksWonBy0: u8,
    sizes: [u8; 4],
}

#[derive(Debug, Clone)]
struct Node {
    visited: u64,
    value: i64,
    state: GameState,
    outcomes: [Option<Box<Node>>; 52],
}

#[wasm_bindgen]
pub struct FullState {
    points: (u8, u8),
    trump: u8,
    player: usize,
    states: [GameState; 4],
    ais: [Option<Node>; 4],
    first: u8,
    trick: u8,
    cards_played: u8,
    card_input: Option<u8>,
}

#[wasm_bindgen]
impl FullState {
    #[wasm_bindgen]
    pub fn new() -> FullState {
        FullState {
            points: (0, 0),
            trump: 0,
            player: 0,
            states: [PLACEHOLDER, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER],
            ais: [None, None, None, None],
            first: 0,
            trick: 0,
            cards_played: 0,
            card_input: None,
        }
    }

    pub fn initGame(&mut self) {
        self.points = (0, 0);
        self.trump = 0;
    }

    pub fn initDeal(&mut self) -> Uint8Array {
        let mut deck: [Card; 52] = (0..52)
            .map(numToCard)
            .collect::<Vec<Card>>()
            .try_into()
            .unwrap();
        self.first = random::<u8>() % 4;
        deck.shuffle(&mut thread_rng());
        let (half1, half2) = deck.split_at(26);
        let (hand0, hand1) = half1.split_at(13);
        let (hand2, hand3) = half2.split_at(13);
        let the_hands: [&[Card]; 4] = [hand0, hand1, hand2, hand3];
        self.states = the_hands
            .iter()
            .enumerate()
            .map(|(i, hand)| {
                let mut opt_hand: [Option<Card>; 13] = [None; 13];
                for (card_i, card) in hand.iter().enumerate() {
                    opt_hand[card_i] = Some(*card);
                }
                GameState {
                    player: self.first,
                    handPlayer: i.try_into().unwrap(),
                    trump: TRUMP_SEQ[self.trump as usize],
                    hand: opt_hand,
                    ruffs: 0,
                    played: 0,
                    trick: [None, None, None],
                    tricksWonBy0: 0,
                    sizes: [13, 13, 13, 13],
                }
            })
            .collect::<Vec<GameState>>()
            .try_into()
            .unwrap();
        self.player = self.first as usize;
        self.ais = [None, None, None, None];
        self.trick = 0;
        self.cards_played = 0;
        let hand: &[u8] = &hand0.iter().map(|x| cardToNum(*x)).collect::<Vec<u8>>();
        hand.into()
    }

    pub fn getTrump(&self) -> u8 {
        self.trump
    }

    pub fn curPlayer(&self) -> u8 {
        self.player as u8
    }

    pub fn curTrick(&self) -> u8 {
        self.trick
    }

    pub fn getNext(&mut self) -> u8 {
        if self.cards_played == 52 {
            return 254;
        }
        let card_played: Card = if self.player == 0 {
            match self.card_input {
                Some(thing) => {
                    self.card_input = None;
                    numToCard(thing)
                }
                None => {
                    return 255;
                }
            }
        } else {
            let mut node: Node = match &self.ais[self.player] {
                None => new_node(self.states[self.player].clone()),
                Some(nnode) => nnode.clone(),
            };
            for _ in 1..10000 {
                let mut hands = hands(&node.state);
                let pathcur: (Vec<usize>, &mut Node) = search(&mut node, &mut hands); //search!
                let path = pathcur.0;
                let cur = pathcur.1;
                let n: i8;
                if cur.state.played.count_ones() == 52 {
                    let tricks0: u8 = cur.state.tricksWonBy0;
                    n = if cur.state.handPlayer == 0 || cur.state.handPlayer == 2 {
                        max(tricks0 as i8 - 6, 0) - max(7 - tricks0 as i8, 0)
                    } else {
                        max(7 - tricks0 as i8, 0) - max(tricks0 as i8 - 6, 0)
                    }
                } else {
                    cur.outcomes[path[path.len() - 1]] = Some(Box::new(new_node(state_transit(
                        &cur.state,
                        numToCard(path[path.len() - 1] as u8),
                    )))); //expand!
                    n = simulate(
                        &(*cur.outcomes[path[path.len() - 1]].as_ref().unwrap()).state,
                        &mut hands,
                    );
                    //simulate!
                }
                propagate(&mut node, path, n);
            }
            let mut highest_value = -4000.0;
            let mut best_move = 100;
            for i in 0..52 {
                match &node.outcomes[i] {
                    None => (),
                    Some(x) => {
                        if x.value as f64 / x.visited as f64 > highest_value {
                            highest_value = x.value as f64 / x.visited as f64;
                            best_move = i;
                        }
                    }
                }
            }
            self.ais[self.player] = Some(node);
            numToCard(best_move as u8)
        };
        for i in 0..4 {
            match &self.ais[i] {
                Some(node) => match &node.outcomes[cardToNum(card_played) as usize] {
                    None => {
                        self.ais[i] = Some(new_node(state_transit(&node.state, card_played)));
                    }
                    Some(x) => {
                        self.ais[i] = Some((**x).clone());
                    }
                },
                None => {}
            }
            self.states[i] = state_transit(&self.states[i], card_played);
        }
        self.player = self.states[0].player as usize;
        if self.states[0].trick[0].is_none() {
            self.trick += 1;
        }
        self.cards_played += 1;
        cardToNum(card_played)
    }

    pub fn playCard(&mut self, card: u8) {
        self.card_input = Some(card);
    }

    pub fn endDeal(&mut self) -> Uint8Array {
        let new_points = if self.states[0].tricksWonBy0 > 6 {
            (self.states[0].tricksWonBy0 - 6, 0)
        } else {
            (0, 7 - self.states[0].tricksWonBy0)
        };

        self.points = (self.points.0 + new_points.0, self.points.1 + new_points.1);
        self.trump += 1;
        self.trump %= 5;
        self.first += 1;
        self.first %= 4;

        let this: &[u8] = &[self.points.0, self.points.1];

        this.into()
    }

    pub fn getTricks(&self) -> u8 {
        self.states[0].tricksWonBy0
    }
}

fn rankOf(c: Card) -> u8 {
    c.rank
}
fn suitOf(c: Card) -> Suit {
    c.suit
}
fn toSuit(n: u8) -> Suit {
    match n {
        0 => Suit::Spades,
        1 => Suit::Hearts,
        2 => Suit::Diamonds,
        3 => Suit::Clubs,
        _ => panic!("ERR in toSuit - it was given too big a number."),
    }
}

fn fromSuit(s: Suit) -> u8 {
    match s {
        Suit::Spades => 0,
        Suit::Hearts => 1,
        Suit::Diamonds => 2,
        Suit::Clubs => 3,
    }
}

fn numToCard(num: u8) -> Card {
    // this assumes the number is legal.
    let rank: u8 = num % 13;
    let suit: Suit = toSuit(num / 13);
    Card { suit, rank }
}

fn cardToNum(card: Card) -> u8 {
    // this assumes card rank is legal.
    let suitN: u8 = fromSuit(suitOf(card));
    suitN * 13 + rankOf(card)
}

fn state_transit(state: &GameState, action: Card) -> GameState {
    // this assumes the action is legal. Please don't do illegal action?
    let hand: [Option<Card>; 13] = if state.player == state.handPlayer {
        let mut new: [Option<Card>; 13] = state.hand;
        for item in &mut new {
            // this assumes action is in hand
            if let Some(x) = item {
                if *x == action {
                    *item = None;
                }
            }
        }
        new
    } else {
        state.hand
    };

    let mut played: u64 = state.played;
    played |= 1 << (cardToNum(action) as u64);

    let ruffs: u16 = match state.trick[0] {
        Some(firstCard) => {
            if suitOf(action) != suitOf(firstCard) {
                let mut new: u16 = state.ruffs;
                new |= 1 << (state.player * 4 + fromSuit(suitOf(firstCard)));
                /*new.set(
                    (state.player * 4 + fromSuit(suitOf(firstCard))).into(),
                    true,
                );*/
                new
            } else {
                state.ruffs
            }
        }
        None => state.ruffs,
    };

    if state.trick[2].is_some() {
        // Ok, new trick after this.

        let firstCard: Card = state.trick[0].unwrap(); //this assumes that tricks is sane
        let mut maxRank: i8 = -1;
        let mut trumped: bool = false;
        let mut winner: u8 = 255;
        for (i, card) in state
            .trick
            .iter()
            .enumerate()
            .chain(iter::once((3, &Some(action))))
        {
            match card {
                Some(x) => {
                    if (suitOf(*x) == suitOf(firstCard) && !trumped && (rankOf(*x) as i8) > maxRank)
                        || (Some(suitOf(*x)) == state.trump
                            && (!trumped || (rankOf(*x) as i8) > maxRank))
                    {
                        maxRank = rankOf(*x) as i8;
                        winner = (8 + i as u8 - (3 - state.player)) % 4;
                        if Some(suitOf(*x)) == state.trump {
                            trumped = true;
                        }
                    }
                }
                None => (),
            }
        }
        let newWon: u8 = state.tricksWonBy0 + if winner == 0 || winner == 2 { 1 } else { 0 };
        let mut sizes = state.sizes;
        sizes[usize::from(state.player)] -= 1;
        GameState {
            player: winner,
            handPlayer: state.handPlayer,
            trump: state.trump,
            hand,
            ruffs,
            played,
            trick: [None, None, None],
            tricksWonBy0: newWon,
            sizes,
        }
    } else {
        let mut trick: [Option<Card>; 3] = state.trick;
        for i in 0..3 {
            if trick[i] == None {
                trick[i] = Some(action);
                break;
            }
        }
        let mut sizes = state.sizes;
        sizes[usize::from(state.player)] -= 1;
        GameState {
            player: (state.player + 1) % 4,
            handPlayer: state.handPlayer,
            trump: state.trump,
            hand,
            ruffs,
            played,
            trick,
            tricksWonBy0: state.tricksWonBy0,
            sizes,
        }
    }
}

const fn card(suit: Suit, rank: u8) -> Card {
    Card { suit, rank }
}

fn new_node(state: GameState) -> Node {
    Node {
        visited: 0,
        value: 0,
        state,
        outcomes: [
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
        ],
    }
}

fn hands(state: &GameState) -> [Vec<u8>; 4] {
    let mut hand: [Vec<u8>; 4] = [vec![], vec![], vec![], vec![]];
    let mut seen = state.played;
    for i in state.hand.iter() {
        match i {
            None => (),
            Some(x) => {
                hand[usize::from(state.handPlayer)].push(cardToNum(*x));
                seen |= 1 << (cardToNum(*x) as u64);
            }
        }
    }
    let mut sizes = state.sizes;
    let mut allowed: [[bool; 4]; 52] = [[true, true, true, true]; 52];
    let mut n_to_assign = 52 - seen.count_ones();
    let mut n_assignable_to = [0, 0, 0, 0];
    for i in 0..52 {
        if (seen & (1 << i)) != 0 {
            continue;
        }
        allowed[i][usize::from(state.handPlayer)] = false;
        for j in 0..4 {
            if state.ruffs & (1 << (j * 4 + i / 13)) != 0 {
                allowed[i][j] = false;
            }
        }
        let mut n = 0;
        let mut last = 0;
        for j in 0..4 {
            if allowed[i][j] {
                n += 1;
                last = j;
                n_assignable_to[j] += 1;
            }
        }
        if n == 1 {
            hand[last].push(i as u8);
            n_to_assign -= 1;
            n_assignable_to[last] -= 1;
            sizes[last] -= 1;
            seen |= (1 as u64) << i;
        }
    }
    let mut running = 0;
    while n_to_assign > 0 {
        let mut any_done = false;

        for i in 0..4 {
            if sizes[i] > 0 && n_assignable_to[i] == sizes[i] {
                for j in 0..52 {
                    if allowed[j][i] && (seen & (1 << j)) == 0 {
                        seen |= 1 << j;
                        hand[i].push(j as u8);
                        sizes[i] -= 1;
                        n_to_assign -= 1;
                        for q in 0..4 {
                            if allowed[j][q] {
                                n_assignable_to[q] -= 1;
                            }
                        }
                    }
                }
                any_done = true;
            }
        }
        if any_done {
            continue;
        }
        while (seen & (1 << running)) != 0 {
            running += 1;
        }
        let mut n: usize = allowed[running]
            .iter()
            .map(|&x| if x { 1 } else { 0 })
            .sum();
        for j in 0..4 {
            if sizes[j] == 0 && allowed[running][j] == true {
                n -= 1;
            }
        }

        for j in 0..4 {
            if !allowed[running][j] || sizes[j] == 0 {
                continue;
            }
            if random::<usize>() % n == 0 {
                hand[j].push(running as u8);
                for i in 0..4 {
                    if allowed[running][i] {
                        n_assignable_to[i] -= 1;
                    }
                }
                sizes[j] -= 1;
                n_to_assign -= 1;
                break;
            } else {
                n -= 1;
            }
        }
        seen = seen | ((1 as u64) << running);
    }
    hand
}

fn search<'a>(node: &'a mut Node, hands: &mut [Vec<u8>; 4]) -> (Vec<usize>, &'a mut Node) {
    let mut cur: &mut Node = node;
    let mut seen: Vec<usize> = vec![];
    loop {
        cur.visited += 1;
        if cur.state.played.count_ones() == 52 {
            // abort
            // no need to do anything with hands
            return (seen, cur);
        }
        let mut score: f64 = -10000.0;
        let mut idx: usize = 100000;
        let mut outin: [bool; 4] = [true; 4];
        for i in hands[usize::from(cur.state.player)].iter() {
            outin[usize::from(i / 13)] = false;
        }
        for i in hands[usize::from(cur.state.player)].iter() {
            match cur.state.trick[0] {
                None => (),
                Some(x) => {
                    if !(fromSuit(suitOf(x)) == i / 13)
                        && !(outin[usize::from(fromSuit(suitOf(x)))])
                    {
                        continue;
                    }
                }
            }
            match &cur.outcomes[usize::from(*i)] {
                None => {
                    seen.push((*i).into());
                    let mut new: Vec<u8> = vec![];
                    for k in &hands[usize::from(cur.state.player)] {
                        if *k != (*i) {
                            new.push(*k)
                        }
                    }
                    hands[usize::from(cur.state.player)] = new;
                    return (seen, cur);
                }
                Some(x) => {
                    let ourscore: f64 = (x.value as f64 / (x.visited as f64 + 1.0))
                        + 6.0 * ((cur.visited as f64).ln() / (x.visited as f64 + 1.0)).sqrt();
                    if ourscore > score {
                        idx = (*i).into();
                        score = ourscore;
                    }
                }
            }
        }
        match &mut cur.outcomes[idx] {
            None => panic!("This will not happen. SEARCH PANIC KW!#"),
            Some(x) => {
                seen.push(idx);

                let mut new: Vec<u8> = vec![];
                for i in &hands[usize::from(cur.state.player)] {
                    if usize::from(*i) != idx {
                        new.push(*i)
                    }
                }

                hands[usize::from(cur.state.player)] = new;
                cur = x; //lol
            }
        }
        //cur.outcomes[idx].clone().unwrap();
        //cur = *(cur.outcomes[idx].unwrap());
    }
}

fn propagate(node: &mut Node, path: Vec<usize>, value: i8) {
    let mut cur: &mut Node = node;
    let mut sgn = -1;
    for i in path {
        cur.value += sgn * i64::from(value);
        sgn = -sgn;
        match &mut cur.outcomes[i] {
            None => panic!("This will not happen. SEARCH PANIC KQ!#"),
            Some(x) => {
                cur = x;
            }
        }
    }
}

fn simulate(state: &GameState, hands: &mut [Vec<u8>; 4]) -> i8 {
    let mut cur: GameState = state.clone();
    loop {
        let tricks0: u8 = cur.tricksWonBy0;
        if cur.played.count_ones() == 52 {
            if cur.handPlayer == 0 || cur.handPlayer == 2 {
                return max(tricks0 as i8 - 6, 0) - max(7 - tricks0 as i8, 0);
            } else {
                return max(7 - tricks0 as i8, 0) - max(tricks0 as i8 - 6, 0);
            }
        }
        let card: Card;
        // ok. find action
        //let hands = hands(&cur);

        let mut outin: [u8; 4] = [0; 4];
        for i in hands[usize::from(cur.player)].iter() {
            outin[usize::from(i / 13)] += 1;
        }
        // i fucked up my life on this one
        let mut n: u8 = cur.sizes[usize::from(cur.player)];
        let mut follow: bool = false;
        let mut led: u8 = 100;
        match cur.trick[0] {
            None => (),
            Some(x) => {
                if outin[usize::from(fromSuit(suitOf(x)))] > 0 {
                    n = outin[usize::from(fromSuit(suitOf(x)))];
                    led = fromSuit(suitOf(x));
                    follow = true;
                }
            }
        }
        let mut p: Card = Card {
            suit: Suit::Diamonds,
            rank: 105,
        }; // see this, something has gone terribly wrong.
        for i in hands[usize::from(cur.player)].iter() {
            if follow && led != i / 13 {
                continue;
            }
            if random::<u8>() % n == 0 {
                p = numToCard(*i);
                let mut new: Vec<u8> = vec![];
                for k in &hands[usize::from(cur.player)] {
                    if *k != *i {
                        new.push(*k)
                    }
                }
                hands[usize::from(cur.player)] = new;

                break;
            } else {
                n -= 1;
            }
        }

        card = p;
        cur = state_transit(&cur, card);
    }
}
