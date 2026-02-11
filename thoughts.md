## Thoughts

The agent is really only learning something, if the return difference after initialisation constantly is positive. This is the actual alpha the agent can generate. Niels suggested to also challenge this with being fully invested. To kind of check against simply buying and holding the selected asset fully as well.



Check the trade execution! Could be some errors here!


WE REALLY NEED TO VERIFY HOW THE BEST MODELS ARE CHOOSEN AND SAVED. Currently both implementations of saa and portfolio allocator use the best mean reward in validation. We should rather use a mixed metric or similar-