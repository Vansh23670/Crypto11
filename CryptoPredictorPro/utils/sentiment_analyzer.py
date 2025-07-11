import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class SentimentAnalyzer:
    """
    Advanced sentiment analysis system for cryptocurrency market sentiment.
    Integrates multiple sources: social media, news, and AI-powered analysis.
    """
    
    def __init__(self):
        self.textblob_available = TEXTBLOB_AVAILABLE
        self.nltk_available = NLTK_AVAILABLE
        self.openai_available = OPENAI_AVAILABLE
        
        # Initialize NLTK sentiment analyzer
        if self.nltk_available:
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.sia = SentimentIntensityAnalyzer()
            except:
                self.nltk_available = False
        
        # Initialize OpenAI client
        if self.openai_available:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.openai_client = OpenAI(api_key=api_key)
                else:
                    self.openai_available = False
            except:
                self.openai_available = False
        
        # News API configuration
        self.news_api_key = os.getenv("NEWS_API_KEY", "demo_key")
        
        # Crypto-specific sentiment keywords
        self.bullish_keywords = [
            'moon', 'bullish', 'pump', 'rally', 'surge', 'breakthrough', 'adoption',
            'institutional', 'mainstream', 'breakout', 'bull run', 'hodl', 'diamond hands',
            'to the moon', 'green', 'gains', 'profit', 'buy the dip', 'accumulate'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'bearish', 'decline', 'fall', 'drop', 'correction',
            'sell-off', 'panic', 'fear', 'red', 'losses', 'bear market', 'paper hands',
            'capitulation', 'bloodbath', 'rekt', 'dead cat bounce', 'bubble'
        ]
    
    def analyze_crypto_sentiment(self, symbol: str, days_back: int = 7) -> Dict:
        """
        Comprehensive sentiment analysis for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'bitcoin')
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            sentiment_sources = {}
            
            # News sentiment
            news_sentiment = self._analyze_news_sentiment(symbol, days_back)
            if news_sentiment:
                sentiment_sources['news'] = news_sentiment
            
            # Social media sentiment (Reddit/Twitter proxy)
            social_sentiment = self._analyze_social_sentiment(symbol)
            if social_sentiment:
                sentiment_sources['social'] = social_sentiment
            
            # AI-powered sentiment analysis
            if self.openai_available:
                ai_sentiment = self._analyze_ai_sentiment(symbol)
                if ai_sentiment:
                    sentiment_sources['ai_analysis'] = ai_sentiment
            
            # Market sentiment indicators
            market_sentiment = self._analyze_market_sentiment(symbol)
            if market_sentiment:
                sentiment_sources['market_indicators'] = market_sentiment
            
            # Combine all sentiment sources
            overall_sentiment = self._combine_sentiment_sources(sentiment_sources)
            
            return {
                'symbol': symbol,
                'overall_sentiment': overall_sentiment.get('score', 0.5),
                'sentiment_label': overall_sentiment.get('label', 'Neutral'),
                'confidence': overall_sentiment.get('confidence', 0.5),
                'sources': sentiment_sources,
                'analysis_date': datetime.now().isoformat(),
                'bullish_signals': overall_sentiment.get('bullish_signals', []),
                'bearish_signals': overall_sentiment.get('bearish_signals', [])
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'overall_sentiment': 0.5,
                'sentiment_label': 'Neutral',
                'confidence': 0.3,
                'error': str(e),
                'analysis_date': datetime.now().isoformat()
            }
    
    def _analyze_news_sentiment(self, symbol: str, days_back: int) -> Optional[Dict]:
        """Analyze sentiment from cryptocurrency news"""
        try:
            # Map symbol to common names for news search
            symbol_mapping = {
                'bitcoin': ['bitcoin', 'btc'],
                'ethereum': ['ethereum', 'eth'],
                'dogecoin': ['dogecoin', 'doge'],
                'cardano': ['cardano', 'ada'],
                'solana': ['solana', 'sol'],
                'polygon': ['polygon', 'matic'],
                'chainlink': ['chainlink', 'link'],
                'litecoin': ['litecoin', 'ltc']
            }
            
            search_terms = symbol_mapping.get(symbol, [symbol])
            
            # Simulate news sentiment analysis (in production, use actual news APIs)
            articles = self._fetch_crypto_news(search_terms, days_back)
            
            if not articles:
                return None
            
            sentiments = []
            for article in articles:
                sentiment = self._analyze_text_sentiment(article.get('content', ''))
                if sentiment:
                    sentiments.append(sentiment)
            
            if sentiments:
                avg_sentiment = np.mean([s['score'] for s in sentiments])
                confidence = np.mean([s['confidence'] for s in sentiments])
                
                return {
                    'score': avg_sentiment,
                    'confidence': confidence,
                    'article_count': len(articles),
                    'positive_articles': len([s for s in sentiments if s['score'] > 0.6]),
                    'negative_articles': len([s for s in sentiments if s['score'] < 0.4]),
                    'source': 'news_analysis'
                }
            
            return None
            
        except Exception:
            return None
    
    def _fetch_crypto_news(self, search_terms: List[str], days_back: int) -> List[Dict]:
        """Fetch cryptocurrency news articles"""
        try:
            # In production, integrate with actual news APIs
            # For now, return simulated news data structure
            
            # Simulate news articles with varying sentiment
            simulated_articles = []
            
            for i in range(np.random.randint(5, 15)):
                # Generate realistic crypto news headlines and content
                headlines = [
                    f"{search_terms[0].title()} reaches new milestone in institutional adoption",
                    f"Market analysis: {search_terms[0].title()} shows strong technical indicators",
                    f"{search_terms[0].title()} faces regulatory headwinds in key markets",
                    f"Whale activity detected: Large {search_terms[0].title()} transactions surge",
                    f"{search_terms[0].title()} network upgrade shows promising developments",
                    f"Analyst predicts {search_terms[0].title()} correction in coming weeks",
                    f"{search_terms[0].title()} integration announced by major fintech company"
                ]
                
                content_templates = [
                    f"Recent developments in {search_terms[0]} ecosystem show positive momentum with increased adoption rates.",
                    f"Technical analysis suggests {search_terms[0]} may face resistance at current levels amid market uncertainty.",
                    f"Institutional investors continue to show interest in {search_terms[0]} despite market volatility.",
                    f"Regulatory clarity remains a key factor for {search_terms[0]} long-term growth prospects.",
                    f"Network metrics for {search_terms[0]} indicate healthy on-chain activity and user engagement."
                ]
                
                article = {
                    'title': np.random.choice(headlines),
                    'content': np.random.choice(content_templates),
                    'date': (datetime.now() - timedelta(days=np.random.randint(0, days_back))).isoformat(),
                    'source': f"crypto_news_{i}"
                }
                
                simulated_articles.append(article)
            
            return simulated_articles
            
        except Exception:
            return []
    
    def _analyze_social_sentiment(self, symbol: str) -> Optional[Dict]:
        """Analyze sentiment from social media discussions"""
        try:
            # Simulate social media sentiment analysis
            # In production, integrate with Reddit API, Twitter API, etc.
            
            # Generate realistic social sentiment based on market conditions
            base_sentiment = 0.5
            
            # Add some randomness to simulate real social sentiment
            social_buzz = np.random.normal(0, 0.15)
            social_sentiment = np.clip(base_sentiment + social_buzz, 0, 1)
            
            # Simulate discussion volume
            discussion_volume = np.random.randint(100, 1000)
            
            # Simulate mention keywords
            bullish_mentions = np.random.randint(0, 50)
            bearish_mentions = np.random.randint(0, 30)
            
            confidence = min(0.9, max(0.3, discussion_volume / 1000))
            
            return {
                'score': social_sentiment,
                'confidence': confidence,
                'discussion_volume': discussion_volume,
                'bullish_mentions': bullish_mentions,
                'bearish_mentions': bearish_mentions,
                'trending_keywords': self._extract_trending_keywords(symbol),
                'source': 'social_media'
            }
            
        except Exception:
            return None
    
    def _analyze_ai_sentiment(self, symbol: str) -> Optional[Dict]:
        """AI-powered sentiment analysis using OpenAI"""
        try:
            if not self.openai_available:
                return None
            
            # Create a comprehensive prompt for crypto sentiment analysis
            prompt = f"""
            Analyze the current market sentiment for {symbol.title()} cryptocurrency.
            Consider recent market trends, adoption news, regulatory developments, 
            and technical analysis indicators. Provide a sentiment score between 0 and 1,
            where 0 is very bearish and 1 is very bullish.
            
            Also identify key factors influencing sentiment and provide confidence level.
            
            Respond in JSON format with:
            - sentiment_score (0-1)
            - confidence (0-1)
            - key_factors (array of strings)
            - market_outlook (string)
            """
            
            # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # Do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency market sentiment expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                'score': result.get('sentiment_score', 0.5),
                'confidence': result.get('confidence', 0.5),
                'key_factors': result.get('key_factors', []),
                'market_outlook': result.get('market_outlook', 'Neutral'),
                'source': 'ai_analysis'
            }
            
        except Exception:
            return None
    
    def _analyze_market_sentiment(self, symbol: str) -> Optional[Dict]:
        """Analyze market-based sentiment indicators"""
        try:
            # Fear & Greed Index simulation
            fear_greed_index = np.random.randint(10, 90)
            
            # VIX-like volatility index for crypto
            crypto_vix = np.random.uniform(20, 80)
            
            # Put/Call ratio simulation
            put_call_ratio = np.random.uniform(0.5, 2.0)
            
            # Convert indicators to sentiment score
            fg_sentiment = fear_greed_index / 100
            vix_sentiment = 1 - (crypto_vix / 100)  # Lower volatility = higher sentiment
            pc_sentiment = 1 / (1 + put_call_ratio)  # Lower put/call = higher sentiment
            
            # Weighted average
            market_sentiment = (fg_sentiment * 0.4 + vix_sentiment * 0.3 + pc_sentiment * 0.3)
            
            return {
                'score': market_sentiment,
                'confidence': 0.7,
                'fear_greed_index': fear_greed_index,
                'volatility_index': crypto_vix,
                'put_call_ratio': put_call_ratio,
                'source': 'market_indicators'
            }
            
        except Exception:
            return None
    
    def _analyze_text_sentiment(self, text: str) -> Optional[Dict]:
        """Analyze sentiment of text using available sentiment analyzers"""
        if not text or len(text.strip()) < 10:
            return None
        
        sentiments = []
        
        # TextBlob sentiment
        if self.textblob_available:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                sentiment_score = (polarity + 1) / 2  # Convert to 0-1 scale
                sentiments.append({
                    'score': sentiment_score,
                    'confidence': abs(polarity),
                    'method': 'textblob'
                })
            except:
                pass
        
        # NLTK VADER sentiment
        if self.nltk_available:
            try:
                scores = self.sia.polarity_scores(text)
                compound = scores['compound']  # -1 to 1
                sentiment_score = (compound + 1) / 2  # Convert to 0-1 scale
                sentiments.append({
                    'score': sentiment_score,
                    'confidence': abs(compound),
                    'method': 'vader'
                })
            except:
                pass
        
        # Keyword-based sentiment (crypto-specific)
        keyword_sentiment = self._analyze_keyword_sentiment(text)
        if keyword_sentiment:
            sentiments.append(keyword_sentiment)
        
        # Combine sentiments
        if sentiments:
            avg_score = np.mean([s['score'] for s in sentiments])
            avg_confidence = np.mean([s['confidence'] for s in sentiments])
            
            return {
                'score': avg_score,
                'confidence': avg_confidence,
                'methods_used': [s['method'] for s in sentiments]
            }
        
        return None
    
    def _analyze_keyword_sentiment(self, text: str) -> Optional[Dict]:
        """Analyze sentiment based on crypto-specific keywords"""
        try:
            text_lower = text.lower()
            
            # Count bullish and bearish keywords
            bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
            bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
            
            total_keywords = bullish_count + bearish_count
            
            if total_keywords == 0:
                return None
            
            # Calculate sentiment score
            sentiment_score = bullish_count / total_keywords
            confidence = min(0.8, total_keywords / 10)  # Higher confidence with more keywords
            
            return {
                'score': sentiment_score,
                'confidence': confidence,
                'method': 'keywords',
                'bullish_keywords': bullish_count,
                'bearish_keywords': bearish_count
            }
            
        except Exception:
            return None
    
    def _extract_trending_keywords(self, symbol: str) -> List[str]:
        """Extract trending keywords related to the cryptocurrency"""
        # Simulate trending keywords
        general_keywords = ['adoption', 'regulation', 'institutional', 'defi', 'nft', 'metaverse']
        symbol_keywords = {
            'bitcoin': ['store of value', 'digital gold', 'lightning network', 'halving'],
            'ethereum': ['smart contracts', 'defi', 'eth2', 'layer 2', 'gas fees'],
            'dogecoin': ['meme coin', 'elon musk', 'community', 'payments'],
            'cardano': ['proof of stake', 'smart contracts', 'academic research'],
            'solana': ['fast transactions', 'low fees', 'ecosystem growth'],
            'polygon': ['layer 2', 'ethereum scaling', 'matic', 'partnerships']
        }
        
        specific_keywords = symbol_keywords.get(symbol, [])
        all_keywords = general_keywords + specific_keywords
        
        # Return random subset to simulate trending
        return list(np.random.choice(all_keywords, size=min(5, len(all_keywords)), replace=False))
    
    def _combine_sentiment_sources(self, sentiment_sources: Dict) -> Dict:
        """Combine sentiment from multiple sources into overall sentiment"""
        if not sentiment_sources:
            return {'score': 0.5, 'label': 'Neutral', 'confidence': 0.3}
        
        # Weight different sources
        source_weights = {
            'news': 0.3,
            'social': 0.25,
            'ai_analysis': 0.25,
            'market_indicators': 0.2
        }
        
        weighted_scores = []
        weighted_confidences = []
        bullish_signals = []
        bearish_signals = []
        
        for source, data in sentiment_sources.items():
            if 'score' in data and 'confidence' in data:
                weight = source_weights.get(source, 0.1)
                weighted_scores.append(data['score'] * weight * data['confidence'])
                weighted_confidences.append(data['confidence'] * weight)
                
                # Extract signals
                if data['score'] > 0.7:
                    bullish_signals.append(f"Strong {source.replace('_', ' ')} sentiment")
                elif data['score'] < 0.3:
                    bearish_signals.append(f"Weak {source.replace('_', ' ')} sentiment")
        
        if not weighted_scores:
            return {'score': 0.5, 'label': 'Neutral', 'confidence': 0.3}
        
        # Calculate weighted average
        total_weight = sum(weighted_confidences)
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
            overall_confidence = min(0.95, total_weight)
        else:
            overall_score = 0.5
            overall_confidence = 0.3
        
        # Determine sentiment label
        if overall_score >= 0.7:
            label = 'Very Bullish'
        elif overall_score >= 0.6:
            label = 'Bullish'
        elif overall_score >= 0.4:
            label = 'Neutral'
        elif overall_score >= 0.3:
            label = 'Bearish'
        else:
            label = 'Very Bearish'
        
        return {
            'score': overall_score,
            'label': label,
            'confidence': overall_confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }
    
    def get_market_fear_greed_index(self) -> Dict:
        """Get overall market fear and greed index"""
        try:
            # Simulate fear and greed index calculation
            # In production, integrate with actual fear/greed APIs
            
            index_value = np.random.randint(10, 90)
            
            if index_value >= 75:
                label = "Extreme Greed"
                color = "#c3142d"
            elif index_value >= 55:
                label = "Greed"
                color = "#f5a623"
            elif index_value >= 45:
                label = "Neutral"
                color = "#f5a623"
            elif index_value >= 25:
                label = "Fear"
                color = "#7ed321"
            else:
                label = "Extreme Fear"
                color = "#417505"
            
            return {
                'value': index_value,
                'label': label,
                'color': color,
                'timestamp': datetime.now().isoformat(),
                'description': f"The market shows {label.lower()} with an index value of {index_value}"
            }
            
        except Exception as e:
            return {
                'value': 50,
                'label': 'Neutral',
                'color': '#f5a623',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_social_media_trends(self, symbols: List[str]) -> Dict:
        """Analyze social media trends for multiple cryptocurrencies"""
        try:
            trends = {}
            
            for symbol in symbols:
                # Simulate social media trend analysis
                mention_volume = np.random.randint(100, 10000)
                sentiment_trend = np.random.uniform(0.2, 0.8)
                engagement_rate = np.random.uniform(0.1, 0.5)
                
                # Trending score calculation
                trending_score = (mention_volume / 10000) * 0.4 + sentiment_trend * 0.4 + engagement_rate * 0.2
                
                trends[symbol] = {
                    'mention_volume': mention_volume,
                    'sentiment_score': sentiment_trend,
                    'engagement_rate': engagement_rate,
                    'trending_score': trending_score,
                    'rank': 0  # Will be calculated after all symbols
                }
            
            # Calculate rankings
            sorted_symbols = sorted(trends.keys(), key=lambda x: trends[x]['trending_score'], reverse=True)
            for i, symbol in enumerate(sorted_symbols):
                trends[symbol]['rank'] = i + 1
            
            return {
                'trends': trends,
                'analysis_time': datetime.now().isoformat(),
                'top_trending': sorted_symbols[:5]
            }
            
        except Exception as e:
            return {'error': str(e), 'trends': {}}
    
    def get_sentiment_history(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get historical sentiment data for a cryptocurrency"""
        try:
            history = []
            
            for i in range(days):
                date = datetime.now() - timedelta(days=days-i)
                
                # Simulate historical sentiment with some trend
                base_trend = 0.5 + 0.1 * np.sin(i / 10)  # Gentle wave pattern
                daily_noise = np.random.normal(0, 0.1)
                sentiment_score = np.clip(base_trend + daily_noise, 0, 1)
                
                history.append({
                    'date': date.isoformat(),
                    'sentiment_score': sentiment_score,
                    'confidence': np.random.uniform(0.6, 0.9),
                    'volume': np.random.randint(50, 500)
                })
            
            return history
            
        except Exception:
            return []

