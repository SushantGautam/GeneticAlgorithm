# Genetic Algorithm
> Genetic Algorithm with floating point gene type in Python.

With options to control console print level. \
**Contributions/ modifications are highly welcomed.**

## Installation
Depends upon numpy for mathematical operations. 
Install numpy with pip.

```sh
pip install numpy
```
You can control what is printed on console with "LogLevel" value at input.json file.\
 **Accepted:** 'ERROR', 'INFO', 'DEBUG'\
'ERROR' : Prints Output Only. \
'INFO' : Print statistics of all generations. \
'DEBUG' : Step by step solution for all generations. 

## Usage example
Navigate to the folder containing main.py.
You can just call the main function and it will do the rest.
```sh
python main.py
```
![image](https://user-images.githubusercontent.com/16721983/94782728-60bb0800-03eb-11eb-9743-337292e9898e.png)


You can customize function at **input.py** and change the GA parameters at **input.json**.

You can control selection method with "SelectionType" value at input.json file.\
 **Accepted:** 'RandomSelection', 'TournamentSelection' \
'RandomSelection' : Random Selection (Default)  \
'TournamentSelection' : Tournament Selection with N/10 members for each group during selection. 

## Meta

Sushant Gautam – [@eSushant](https://twitter.com/eSushant) – susant.gautam@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/SushantGautam/GeneticAlgorithm/](https://github.com/sushantgautam/)

## Contributing

1. Fork it (<https://github.com/SushantGautam/GeneticAlgorithm/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->

