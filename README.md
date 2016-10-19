Q Learning Algorithm for Rust
==========

The library itself is in a very early stage.

Currently it only has a naive implemention, of which every action is deterministically lead to a single state, rather than
a distribution of states, e.g. Multinomial.

TODO: 
- [ ] Documentation.
- [ ] Convergence test.
- [ ] Support for a distribution of action result.
- [ ] Publish to crates.io