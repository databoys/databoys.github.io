---
layout: post
title: Bald Eagle PVA
published: true
---

Population Viability Analysis, or PVA, is a core concept in conservation. As someone who loves birds and data I wanted to explore using Lefkovitch Matrix Models in a PVA on hypothetical population of Bald Eagles.

To start with, Lefkovitch Matrix Models are a modified Leslie Matrix Model. Leslie Models are commonly used as a framework to study population vialibity. They can be manipulated (and thus are not true Leslie Matrix Models anymore) to include density dependence - parameters are functions of the population size. Generally, in a Leslie Model a population of some species is broken down into age classes, say, age 0, 1, 2, ... n. These models are similar to discrete-time Markov Chains. These classes are broken down into a square matrix. The following is needed for a Leslie Model:

> n = the count of individuals in each age class

> s = the fraction of individuals that survive from the previous age class

> f = fecundity -> per capita average number of offspring reaching adulthood.

> f = (s)(b) where b is the number of offspring produced at the next age class

> ...add some magic (will show below)...

> [n t+1] = [Leslie Matrix][n]

Now, there are some BIG generalizations in this model. First and foremost, it accounts only for females (making offsprings). Initially, this model requires a ton of information to be obtained for the population. THe Lefkovitch Matrix Model accounts for one of these requirements: being able to identify every age of individuals. Moving on to the birds-

Often it is difficult to age Bald Eagles. However, you may be able to tell the general age group. The following is a generalization of this problem and has some wonky fake data...

Let's say you can tell what a first year Bald Eagle chick (year 0) looks like. You can also identify a second year (year 1). But suddenly you cannot tell the difference between a year 2, 3, or 4. Let's call this whole group, or stage, 'immatures'. Finally, full blown adult plumage birds all look the same. Since Bald Eagles may live to be 20, year 5 - 20 are 'adults'.

We have: 

> year 0

> year 1

> immatures (year 2- 4)

> adults (years 5 - 20)

Instead of age classes, Lefkovitch Matrix Models use stage classes. They just group things differently! We will use the following information for our model:  

#1. number of individuals within each age stage: 

> n0 = 100 (100 year 0)

> n1 = 50

> n2-4 = 61

> n5-20 = 189

#2. probability of surviving each age:

> L0 = 0.5 (50% survival rate in year 0)

> L1 = 0.5 

> L2-4 (or Limmature) = 0.8

> L5-20 (or Ladult) = 0.95

#3. number of offspring produced on average per stage

> m0 = 0

> m1 = 0

> m2-4 = 0.33 (year 4 usually have an offspring, thus 1/3)

> m5-20 = 2

#4. probability of surviving into next stage (note: not within stage...)

> p0 = 0.5 (survive year 0 to year 1)

> p1 = 0.5 (survive year 1 to year 2)

> p2 = 0.8 (survive year 4 to year 5)

#5. probability of surviving within immature and adult stages

G(imm) = sum all individual immatures from ages 2 to 3 * probability of surviving each age over all ages 2 to 4 * probability of surviving each age. This accounts for year 2 becoming year 3 and year 3 becoming year 4 but NOT year 4 becoming year 5 (that is a stage change).

G(adult) = same thing for the adult stage. Sum ages 5 to 19 * prob / ages 5 to 20 * prob. 
 
Since roughly 80% of the birds in years 2 - 4 survive, and we have a starting immature size of 61 birds, we can say there are 45 year 2 and 3 (25 + 20 where 20 is 80% of 25) and 16 year 4 (where 16 is 80% of 20). Thus, 25 + 20 + 16 = 61, and 61 is the number of immature birds. 

> G(imm) = (25 * 0.8) + (20 * 0.8) / (25 * 0.8) + (20 * 0.8) + (16 * 0.95)

> G(imm) = 0.71

Note: year 4 survival rate to year 5 (adult) is 95%. 

Since roughly 95% of the birds in years 5 - 20 survive...
> G(adult) = (15 * 0.95) + (14 * 0.95) ... / (15 * 0.95) + (14 * 0.95) ... (10 * 0.1)

> G(adult) = 0.99

Note: the last in the denominator (10 * 0.1) was because it is speculated that some Bald Eagles live past 20 years of age. Thus, 10% of the 10 20 year olds in this model survive to the next year. This, of course, leaves out any bird older than 21. 

#6. Lastly, F values for weighted fecundity: 

> Fi = (P i - 1)(mi)

> F1 = (0.0)(0) = 0

> F2 = (0.5)(0) = 0 

> F3 = (0.5)(0.33) = 0.165

> F4 = (0.95)(2) = 1.9

Now for the matrix! 

> Lefkovitch = [ F1 F2 F3 F4; p0 0 0 0; 0 p1 G(imm) 0; 0 0 p2 G(adult)]

> n (current population) = [year0; year1; immature; adult]

> [Lefk][n (current population)]

>[0 0 0.165 1.9; 0.5 0 0 0; 0 0.5 0.7 0; 0 0 0.8 0.99][100; 50; 61; 189]

>= n (next year) = [369; 50; 68; 237]

Well there you have it. In one year the current population size should be increased to 369 year 0 birds, 50 year 1 birds, 68 year 2 - 4 (immature) birds, and 237 year 5 - 20 (adult) birds. To complete this model, you would run this through a number of years to see the increase of decrease of the population. Also, running the model multiple times with stochastic parameters can account for random events and give an approximation for the chance of population survival, thus, population viability. 

