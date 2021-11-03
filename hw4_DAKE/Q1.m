clear; close all; clc;

%%
% Let B be the dominant allele that codes for brown eyes and b be the
% recessive allele that codes for blue eyes. We are given that the male
% chimpanzee has blue eyes. Hence the genotype of the male chimpanzee is
% bb. The female chimpanzee has brown eyes and hence she can be either BB
% or Bb genotypically. We are given that both of these are equally likely.
% For the sake of consistency, let's keep prefixes of m for male, f for 
% female and c for child at the end of genotypes.
%%
% Hence we have:
%%
% P(bbm) = 1; P(BBf) = P(Bbf) = 0.5;
%%
% We are also given that the a priori probabilities of the child having any
% of the four possible combinations of genotypes is the same. Hence we
% have:
%%
% P(BBc) = P(Bbc) = P(bBc) = P(bbc) = 0.25
%%
% Since Bbc and bBc are the exact same genotypes, we have:
% P(BBc) = P(bbc) = 0.25; P(Bbc) = 0.5;

%% a)
% We are given that the child has brown eyes and we are to compute the
% likelihood that the female chimp has a blue-eyed gene i.e. that the
% genotype of the female chimp is Bb. Since the male chimp is genotypically
% bb, the phenotype of the child is determined by which allele it receives
% from the mother. If it receives allele B, then the genotype of the child
% becomes Bb and the phenotype is brown eyes. On the other hand, if it
% receives allele b, then the genotype of the child becomes bb and the
% phenotype is blue eyes. Hence the desired probability is:
%%
% $P(Bbm|Bc)$
%%
% Using Bayes' rule, we have:
%%
% $P(Bbm|Bc) = \frac{P(Bc|Bbm)P(Bbm)}{P(Bc)}$
%%
% The probability of the child getting B allele if the mother is
% genotypically Bb is 1/2 and the child will have allele B if the genotype
% of the child is either BB or Bb. Hence P(Bc) = 1/4 + 1/2 = 3/4
%%
% Substituting the values, we get:
%%
% $P(Bbm|Bc) = \frac{1/2\times 1/2}{3/4}$
%%
% $P(Bbm|Bc) = \frac{1}{3}$

%% b)
% Let B1 be the event that the first child has brown eyes and B2 be the 
% event that the second child has brown eyes. Hence we are interested in
% computing:
%%
% $P(Bbm|B1c \cap B2c)$
%%
% Using Bayes' rule, we have:
%%
% $P(Bbm|B1c \cap B2c) = \frac{P(B1c \cap B2c|Bbm)P(Bbm)}{P(B1c \cap B2c)}$
%%
% The probability of one child having brown eyes is independent of the
% other. Hence we have:
%%
% $P(B1c \cap B2c) = P(B1c) \times P(B2c)$
%%
% And we also have:
%%
% $P(B1c \cap B2c|Bbm) = P(B1c|Bbm) \times P(B2c|Bbm)$
%%
% Substituting this in the formula, we get:
%%
% $P(Bbm|B1c \cap B2c) = \frac{P(B1c|Bbm) \times P(B2c|Bbm)\times P(Bbm)}{P(B1c) \times P(B2c)}$
%%
% Substituting the values, we get:
%%
% $P(Bbm|B1c \cap B2c) = \frac{1/2 \times 1/2\times 1/2}{3/4 \times 3/4}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c) = \frac{2}{9}$

%% c)
% Instead of 2 children, if there were N children with brown eyes, then
% using the same logic as in (b), we have:
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap BNc) = \frac{P(B1c \cap B2c \cap ... \cap BNc|Bbm)P(Bbm)}{P(B1c \cap B2c \cap ... \cap BNc)}$
%%
% The probability of one child having brown eyes is independent of the
% other. Hence we have:
%%
% $P(B1c \cap B2c \cap ... \cap BNc) = P(B1c) \times P(B2c) \times ... \times P(BNc)$
%%
% And we also have:
%%
% $P(B1c \cap B2c \cap ... \cap BNc|Bbm) = P(B1c|Bbm) \times P(B2c|Bbm) \times ... \times P(BNc|Bbm)$
%%
% Since $$P(B1c) = P(B2c) = ... = P(BNc) = P(Bc) $$, and $$P(B1c|Bbm) =
% P(B2c|Bbm) = ... = P(BNc|Bbm) = P(Bc|Bbm) $$,
%%
% We have:
%%
% $P(B1c \cap B2c \cap ... \cap BNc) = [P(Bc)]^N$
%%
% and
%%
% $P(B1c \cap B2c \cap ... \cap BNc|Bbm) = [P(Bc|m)]^N$
%%
% Substituting this back into the formula, we get:
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap BNc) = \frac{[P(Bc|Bbm)]^N\times P(Bbm)}{[P(Bc)]^N}$
%%
% Therefore,
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap BNc) = \frac{P(Bc|Bbm)}{P(Bc)}^N\times P(Bbm)$
%%
% Because $$P(Bc|Bbm) = 1/2 $$, $$P(Bc) = 3/4 $$, and $$P(Bbm) = 1/2 $$, we
% have:
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap BNc) = \frac{1/2}{3/4}^N\times 1/2$
%%
% $P(Bbm|B1c \cap B2c \cap ... \cap BNc) = \frac{2}{3}^N\times \frac{1}{2}$
