# Algorithm Guidelines

## Prediction Algorithm Standards

### Algorithm Categories

#### Traditional Algorithms (6 types)
- **Frequency Analysis**: Based on historical appearance frequency
- **Hot/Cold Analysis**: Recent performance vs. average frequency
- **Missing Analysis**: Based on number absence periods with compensation logic
- **Markov Chain**: State transition probability matrices
- **Bayesian Analysis**: Prior/posterior probability with likelihood functions
- **Correlation Analysis**: Inter-number relationship and dependency patterns

#### 9 Mathematical Models
1. **Statistical Analysis**: Descriptive stats, distribution analysis, variance, skewness, kurtosis
2. **Probability Theory**: Conditional, joint, marginal probabilities, independence testing
3. **Frequency Pattern Analysis**: Frequency cycles, pattern sequences, trend analysis
4. **Decision Tree Analysis**: Sum-based and odd/even ratio decision rules
5. **Periodicity Analysis**: Weekly, monthly, seasonal, numerical cycle patterns
6. **Historical Association**: Time-lag correlation, sequence correlation, pattern correlation
7. **Enhanced Markov Chain**: Multi-order chains, transition matrices, prediction probabilities
8. **Enhanced Bayesian**: Prior distributions, likelihood functions, posterior updates
9. **Regression Analysis**: Linear trends, polynomial fitting, time series, moving averages

#### Advanced Integration Algorithms (5 types)
- **Markov-Bayesian Fusion**: 60% Markov + 40% Bayesian weighted combination
- **Hot/Cold-Markov Integration**: 40% hot/cold + 60% Markov state transitions
- **Multi-dimensional Probability**: 25% each of frequency, missing, Markov, Bayesian
- **Comprehensive Weight Scoring**: 6-dimension scoring with dynamic weights
- **Advanced Pattern Recognition**: Consecutive numbers, sums, odd/even, size patterns

#### Machine Learning Algorithms (4 types)
- **LSTM Deep Learning**: Time series prediction with 20-period sequences
- **Monte Carlo Simulation**: Probability distribution-based random simulation
- **Clustering Analysis**: K-Means, GMM, DBSCAN pattern recognition
- **Super Predictor**: Ensemble of multiple ML algorithms

#### Adaptive Learning Algorithms (3 types)
- **Multi-Armed Bandit**: UCB1, Epsilon-Greedy, Thompson Sampling
- **Reinforcement Learning**: Dynamic weight adjustment, continuous optimization
- **Ensemble Learning**: Intelligent fusion of multiple algorithms

### Algorithm Implementation Standards

#### Prediction Method Signatures
```python
def predict_method(self, count: int = 5, periods: int = 500) -> List[Tuple[List[int], List[int]]]:
    """
    Standard prediction method signature
    
    Args:
        count: Number of predictions to generate (1-20)
        periods: Historical periods to analyze (100-3000)
    
    Returns:
        List of tuples: [(front_balls, back_balls), ...]
        front_balls: List of 5 integers (1-35)
        back_balls: List of 2 integers (1-12)
    """
```

#### Compound Betting Standards
- **Front Area**: 6-15 numbers supported
- **Back Area**: 3-12 numbers supported
- **Combination Calculation**: C(n,5) × C(m,2)
- **Cost Calculation**: combinations × 3 yuan per bet

#### Confidence Scoring
- Use 0.0-1.0 scale for prediction confidence
- Higher scores indicate stronger algorithmic consensus
- Include uncertainty quantification for Bayesian methods

### Performance Metrics
- **Hit Rate**: Percentage of periods with any number matches
- **Accuracy**: Weighted score based on match count (2+0, 1+1, 0+2, etc.)
- **Consistency**: Standard deviation of prediction performance
- **Adaptability**: Performance improvement over time with learning

### Algorithm Selection Guidelines
- **Conservative Users**: 9 mathematical models, Markov chains, comprehensive scoring
- **Aggressive Users**: Advanced integration, adaptive learning, super predictor
- **Balanced Users**: Markov-Bayesian fusion, multi-dimensional probability analysis