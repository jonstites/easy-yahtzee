# Optimal Yahtzee

[![Build Status](https://travis-ci.com/jonstites/sir-rolls-a-lot.svg?branch=master)](https://travis-ci.com/jonstites/sir-rolls-a-lot)

## Yahtzee

Yahtzee is a dice-based game that contains elements of both luck and strategy.

[Wikipedia (CC BY-SA 3.0)](https://en.wikipedia.org/wiki/Yahtzee) has an excellent summary.

> The objective of the game is to score points by rolling five dice to make certain combinations. The dice can be rolled up to three times in a turn to try to make various scoring combinations and dice must remain in the box. A game consists of thirteen rounds. After each round the player chooses which scoring category is to be used for that round. Once a category has been used in the game, it cannot be used again. The scoring categories have varying point values, some of which are fixed values and others for which the score depends on the value of the dice. A Yahtzee is five-of-a-kind and scores 50 points, the highest of any category. The winner is the player who scores the most points. 

## Literature Search


Novel: nothing? Maybe correctness / exactness?
Performance?

buy that Yahtzee book?

## Design
Object: maximize expected score.

Goals:
correctness (what does this mean?)
performance (what kinds?)
security (...?)
ease-of-use

non-goals:
pretty?...


### User interaction

There are many possibilities for allowing a user to interact with this project.

1. Command-line usage only.
2. Some kind of native GUI.
3. Some kind of phone-based app.
4. Via web-browser that queries a running server.
5. Via web-browser that performs calculations on the client-side

#### Command-line usage

Pros: 
- simple design
- very low overhead, small binaries

Cons:
- not very ergonomic


#### Native GUI

Pros:
- very flexible UI

Cons:
- very complicated
- dramatically larger binaries, most likely

#### App

Pros:
- convenience
- does this even exist?

Cons:
- complicated
- distraction from the fun/interesting bits

#### Web-browser, server-side calculations

Pros:
- extremely convenient
- can even be done without javascript
- high flexibility with UI

Cons:
- network connection required
- I'd have to have a server 24/7, or use some kind of Cloud Function
- bloated binaries?

#### Web-browser, client-side calculations

Pros:
- extremely convenient
- once code is cached, no network connection required
- faster than server-side?
- fun with WebAssembly?
- very high flexibility with UI

Cons:
- how much bandwidth required for loading initial values?
- javascript
- bloated binaries?

#### Conclusion

I'm just going to reject the phone-app and native GUI. I'm not particularly interested in learning either one at the moment, so it just seems like a lot more work than necessary. It would distract from the primary goals of this project.

At a minimum, there _should_ be a terminal-only interface. This should be compatible with a very stripped-down, small binary, and it should be fast.

Ideas - either it could be command-line argument only, or it could use ncurses, or both. Empirically, will need to see what kind of binary size is required for supporting ncurses vs command line arguments.

Probably there should _also_ be a browser version too. This makes it convenient for e.g. potential employers to verify that the thing works, to try it out, etc. And of course, there are more people who use browsers than terminals.

Client vs server. 

So I think it will come down to an empirical question - how fast can the client-side be? How fast is the server-side version? How big are the files that need to be loaded on the client side?

Also, I'd need to figure out where I would even put server-side code...

I'll have to try it out.

Note: regarding sending the expected values to the client. It seems like there's potential for compressing the values down. Maybe the same answers could be gotten from 8 bit values, rather than 32-bit floats or 64-bit floats. This question of lossy compression of numbers that have a very non-random distribution seems a little like image compression - maybe there is something interesting going on there.

## Math