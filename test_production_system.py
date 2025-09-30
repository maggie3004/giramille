<<<<<<< HEAD
"""
Test Production System for Giramille Style
Comprehensive testing of color accuracy, performance, and quality
"""

import requests
import json
import base64
import time
from PIL import Image
import io
import os
from typing import Dict, List

class ProductionSystemTester:
    """Test the production system comprehensively"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_color_accuracy(self):
        """Test color accuracy with specific prompts"""
        print("🎨 Testing Color Accuracy...")
        
        color_tests = [
            {
                "prompt": "blue house with red car, cartoon style",
                "expected_colors": ["blue", "red"],
                "description": "Blue house with red car"
            },
            {
                "prompt": "green tree with yellow flowers, nature scene",
                "expected_colors": ["green", "yellow"],
                "description": "Green tree with yellow flowers"
            },
            {
                "prompt": "purple mountain with orange sunset, landscape",
                "expected_colors": ["purple", "orange"],
                "description": "Purple mountain with orange sunset"
            },
            {
                "prompt": "pink house with blue windows, cute style",
                "expected_colors": ["pink", "blue"],
                "description": "Pink house with blue windows"
            },
            {
                "prompt": "yellow car in front of white house, sunny day",
                "expected_colors": ["yellow", "white"],
                "description": "Yellow car with white house"
            }
        ]
        
        for i, test in enumerate(color_tests, 1):
            print(f"\n📸 Test {i}: {test['description']}")
            
            try:
                # Generate image
                result = self._generate_image(test['prompt'], quality="high")
                
                if result['success']:
                    # Analyze colors
                    color_accuracy = self._analyze_color_accuracy(
                        test['prompt'], 
                        result['image'], 
                        test['expected_colors']
                    )
                    
                    # Save test image
                    filename = f"color_test_{i}_{test['description'].replace(' ', '_')}.png"
                    self._save_image(result['image'], filename)
                    
                    test_result = {
                        'test': test['description'],
                        'prompt': test['prompt'],
                        'expected_colors': test['expected_colors'],
                        'color_accuracy': color_accuracy,
                        'generation_time': result['generation_time'],
                        'success': True,
                        'filename': filename
                    }
                    
                    print(f"✅ Color Accuracy: {color_accuracy:.2f}")
                    print(f"⏱️  Generation Time: {result['generation_time']:.2f}s")
                    
                else:
                    test_result = {
                        'test': test['description'],
                        'prompt': test['prompt'],
                        'error': result['error'],
                        'success': False
                    }
                    print(f"❌ Error: {result['error']}")
                
                self.test_results.append(test_result)
                
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                self.test_results.append({
                    'test': test['description'],
                    'prompt': test['prompt'],
                    'error': str(e),
                    'success': False
                })
    
    def test_performance(self):
        """Test system performance"""
        print("\n⚡ Testing Performance...")
        
        # Test different quality settings
        quality_tests = [
            {"quality": "fast", "expected_time": 10},
            {"quality": "balanced", "expected_time": 20},
            {"quality": "high", "expected_time": 40}
        ]
        
        test_prompt = "blue house with red car, cartoon style"
        
        for test in quality_tests:
            print(f"\n🔧 Testing {test['quality']} quality...")
            
            try:
                start_time = time.time()
                result = self._generate_image(test_prompt, quality=test['quality'])
                end_time = time.time()
                
                actual_time = end_time - start_time
                
                test_result = {
                    'quality': test['quality'],
                    'expected_time': test['expected_time'],
                    'actual_time': actual_time,
                    'within_expected': actual_time <= test['expected_time'],
                    'success': result['success']
                }
                
                if result['success']:
                    print(f"✅ Time: {actual_time:.2f}s (Expected: {test['expected_time']}s)")
                    if actual_time <= test['expected_time']:
                        print("✅ Performance: GOOD")
                    else:
                        print("⚠️  Performance: SLOW")
                else:
                    print(f"❌ Failed: {result['error']}")
                
                self.test_results.append(test_result)
                
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                self.test_results.append({
                    'quality': test['quality'],
                    'error': str(e),
                    'success': False
                })
    
    def test_system_health(self):
        """Test system health and metrics"""
        print("\n🏥 Testing System Health...")
        
        try:
            # Test basic health
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Basic Health: {health_data['status']}")
                print(f"📊 Model Loaded: {health_data['model_loaded']}")
            else:
                print(f"❌ Basic Health Failed: {response.status_code}")
            
            # Test production health
            response = requests.get(f"{self.base_url}/api/production/health")
            if response.status_code == 200:
                prod_health = response.json()
                print(f"✅ Production Health: {prod_health['status']}")
                print(f"💾 Cache Available: {prod_health['cache_available']}")
                print(f"🖥️  CPU: {prod_health['system']['cpu_percent']:.1f}%")
                print(f"🧠 Memory: {prod_health['system']['memory_percent']:.1f}%")
            else:
                print(f"❌ Production Health Failed: {response.status_code}")
            
            # Test metrics
            response = requests.get(f"{self.base_url}/api/production/metrics")
            if response.status_code == 200:
                metrics = response.json()
                print(f"📈 Total Requests: {metrics['requests_total']}")
                print(f"✅ Success Rate: {metrics['requests_success']}/{metrics['requests_total']}")
                print(f"⏱️  Avg Generation Time: {metrics['avg_generation_time']:.2f}s")
                print(f"💾 Cache Hit Rate: {metrics['cache_hits']}/{metrics['cache_hits'] + metrics['cache_misses']}")
            else:
                print(f"❌ Metrics Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Health Check Exception: {str(e)}")
    
    def _generate_image(self, prompt: str, quality: str = "balanced") -> Dict:
        """Generate image using API"""
        try:
            response = requests.post(f"{self.base_url}/api/generate", json={
                'prompt': prompt,
                'style': 'png',
                'quality': quality
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    # Decode base64 image
                    image_data = data['image'].split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    
                    return {
                        'success': True,
                        'image': image_bytes,
                        'generation_time': 0  # Will be calculated by caller
                    }
                else:
                    return {
                        'success': False,
                        'error': data.get('error', 'Unknown error')
                    }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_color_accuracy(self, prompt: str, image_bytes: bytes, expected_colors: List[str]) -> float:
        """Analyze color accuracy of generated image"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB
            image = image.convert('RGB')
            
            # Get dominant colors
            colors = image.getcolors(maxcolors=256*256*256)
            if not colors:
                return 0.0
            
            # Sort by frequency
            colors.sort(key=lambda x: x[0], reverse=True)
            
            # Convert to color names
            dominant_colors = []
            for count, (r, g, b) in colors[:10]:  # Top 10 colors
                color_name = self._rgb_to_color_name(r, g, b)
                if color_name:
                    dominant_colors.append(color_name)
            
            # Calculate accuracy
            matches = 0
            for expected_color in expected_colors:
                if expected_color in dominant_colors:
                    matches += 1
            
            return matches / len(expected_colors) if expected_colors else 0.0
            
        except Exception as e:
            print(f"Color analysis error: {e}")
            return 0.0
    
    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """Convert RGB to color name"""
        color_ranges = {
            'red': [(150, 0, 0), (255, 150, 150)],
            'blue': [(0, 0, 150), (150, 150, 255)],
            'green': [(0, 150, 0), (150, 255, 150)],
            'yellow': [(150, 150, 0), (255, 255, 150)],
            'purple': [(150, 0, 150), (255, 150, 255)],
            'orange': [(255, 165, 0), (255, 200, 100)],
            'pink': [(255, 192, 203), (255, 220, 220)],
            'brown': [(139, 69, 19), (200, 150, 100)],
            'white': [(200, 200, 200), (255, 255, 255)],
            'black': [(0, 0, 0), (100, 100, 100)]
        }
        
        for color_name, (min_rgb, max_rgb) in color_ranges.items():
            if (min_rgb[0] <= r <= max_rgb[0] and 
                min_rgb[1] <= g <= max_rgb[1] and 
                min_rgb[2] <= b <= max_rgb[2]):
                return color_name
        
        return None
    
    def _save_image(self, image_bytes: bytes, filename: str):
        """Save image to file"""
        try:
            with open(filename, 'wb') as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # Color accuracy results
        color_tests = [r for r in self.test_results if 'color_accuracy' in r]
        if color_tests:
            avg_accuracy = sum(r['color_accuracy'] for r in color_tests) / len(color_tests)
            print(f"\n🎨 Color Accuracy: {avg_accuracy:.2f} ({avg_accuracy*100:.1f}%)")
            
            for test in color_tests:
                status = "✅" if test['color_accuracy'] > 0.7 else "⚠️" if test['color_accuracy'] > 0.4 else "❌"
                print(f"  {status} {test['test']}: {test['color_accuracy']:.2f}")
        
        # Performance results
        perf_tests = [r for r in self.test_results if 'quality' in r]
        if perf_tests:
            print(f"\n⚡ Performance Results:")
            for test in perf_tests:
                if test['success']:
                    status = "✅" if test['within_expected'] else "⚠️"
                    print(f"  {status} {test['quality']}: {test['actual_time']:.2f}s (Expected: {test['expected_time']}s)")
                else:
                    print(f"  ❌ {test['quality']}: Failed - {test.get('error', 'Unknown error')}")
        
        # Overall success rate
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get('success', False)])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📈 Overall Success Rate: {success_rate:.2f} ({success_rate*100:.1f}%)")
        print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
        
        # Save detailed report
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'color_accuracy': avg_accuracy if color_tests else 0,
            'test_results': self.test_results
        }
        
        with open('test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: test_report.json")
        print("="*60)

def main():
    """Run comprehensive production system tests"""
    print("🚀 Giramille Production System Tester")
    print("="*60)
    
    tester = ProductionSystemTester()
    
    # Run all tests
    tester.test_system_health()
    tester.test_color_accuracy()
    tester.test_performance()
    
    # Generate report
    tester.generate_report()
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
=======
"""
Test Production System for Giramille Style
Comprehensive testing of color accuracy, performance, and quality
"""

import requests
import json
import base64
import time
from PIL import Image
import io
import os
from typing import Dict, List

class ProductionSystemTester:
    """Test the production system comprehensively"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
    
    def test_color_accuracy(self):
        """Test color accuracy with specific prompts"""
        print("🎨 Testing Color Accuracy...")
        
        color_tests = [
            {
                "prompt": "blue house with red car, cartoon style",
                "expected_colors": ["blue", "red"],
                "description": "Blue house with red car"
            },
            {
                "prompt": "green tree with yellow flowers, nature scene",
                "expected_colors": ["green", "yellow"],
                "description": "Green tree with yellow flowers"
            },
            {
                "prompt": "purple mountain with orange sunset, landscape",
                "expected_colors": ["purple", "orange"],
                "description": "Purple mountain with orange sunset"
            },
            {
                "prompt": "pink house with blue windows, cute style",
                "expected_colors": ["pink", "blue"],
                "description": "Pink house with blue windows"
            },
            {
                "prompt": "yellow car in front of white house, sunny day",
                "expected_colors": ["yellow", "white"],
                "description": "Yellow car with white house"
            }
        ]
        
        for i, test in enumerate(color_tests, 1):
            print(f"\n📸 Test {i}: {test['description']}")
            
            try:
                # Generate image
                result = self._generate_image(test['prompt'], quality="high")
                
                if result['success']:
                    # Analyze colors
                    color_accuracy = self._analyze_color_accuracy(
                        test['prompt'], 
                        result['image'], 
                        test['expected_colors']
                    )
                    
                    # Save test image
                    filename = f"color_test_{i}_{test['description'].replace(' ', '_')}.png"
                    self._save_image(result['image'], filename)
                    
                    test_result = {
                        'test': test['description'],
                        'prompt': test['prompt'],
                        'expected_colors': test['expected_colors'],
                        'color_accuracy': color_accuracy,
                        'generation_time': result['generation_time'],
                        'success': True,
                        'filename': filename
                    }
                    
                    print(f"✅ Color Accuracy: {color_accuracy:.2f}")
                    print(f"⏱️  Generation Time: {result['generation_time']:.2f}s")
                    
                else:
                    test_result = {
                        'test': test['description'],
                        'prompt': test['prompt'],
                        'error': result['error'],
                        'success': False
                    }
                    print(f"❌ Error: {result['error']}")
                
                self.test_results.append(test_result)
                
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                self.test_results.append({
                    'test': test['description'],
                    'prompt': test['prompt'],
                    'error': str(e),
                    'success': False
                })
    
    def test_performance(self):
        """Test system performance"""
        print("\n⚡ Testing Performance...")
        
        # Test different quality settings
        quality_tests = [
            {"quality": "fast", "expected_time": 10},
            {"quality": "balanced", "expected_time": 20},
            {"quality": "high", "expected_time": 40}
        ]
        
        test_prompt = "blue house with red car, cartoon style"
        
        for test in quality_tests:
            print(f"\n🔧 Testing {test['quality']} quality...")
            
            try:
                start_time = time.time()
                result = self._generate_image(test_prompt, quality=test['quality'])
                end_time = time.time()
                
                actual_time = end_time - start_time
                
                test_result = {
                    'quality': test['quality'],
                    'expected_time': test['expected_time'],
                    'actual_time': actual_time,
                    'within_expected': actual_time <= test['expected_time'],
                    'success': result['success']
                }
                
                if result['success']:
                    print(f"✅ Time: {actual_time:.2f}s (Expected: {test['expected_time']}s)")
                    if actual_time <= test['expected_time']:
                        print("✅ Performance: GOOD")
                    else:
                        print("⚠️  Performance: SLOW")
                else:
                    print(f"❌ Failed: {result['error']}")
                
                self.test_results.append(test_result)
                
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
                self.test_results.append({
                    'quality': test['quality'],
                    'error': str(e),
                    'success': False
                })
    
    def test_system_health(self):
        """Test system health and metrics"""
        print("\n🏥 Testing System Health...")
        
        try:
            # Test basic health
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Basic Health: {health_data['status']}")
                print(f"📊 Model Loaded: {health_data['model_loaded']}")
            else:
                print(f"❌ Basic Health Failed: {response.status_code}")
            
            # Test production health
            response = requests.get(f"{self.base_url}/api/production/health")
            if response.status_code == 200:
                prod_health = response.json()
                print(f"✅ Production Health: {prod_health['status']}")
                print(f"💾 Cache Available: {prod_health['cache_available']}")
                print(f"🖥️  CPU: {prod_health['system']['cpu_percent']:.1f}%")
                print(f"🧠 Memory: {prod_health['system']['memory_percent']:.1f}%")
            else:
                print(f"❌ Production Health Failed: {response.status_code}")
            
            # Test metrics
            response = requests.get(f"{self.base_url}/api/production/metrics")
            if response.status_code == 200:
                metrics = response.json()
                print(f"📈 Total Requests: {metrics['requests_total']}")
                print(f"✅ Success Rate: {metrics['requests_success']}/{metrics['requests_total']}")
                print(f"⏱️  Avg Generation Time: {metrics['avg_generation_time']:.2f}s")
                print(f"💾 Cache Hit Rate: {metrics['cache_hits']}/{metrics['cache_hits'] + metrics['cache_misses']}")
            else:
                print(f"❌ Metrics Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Health Check Exception: {str(e)}")
    
    def _generate_image(self, prompt: str, quality: str = "balanced") -> Dict:
        """Generate image using API"""
        try:
            response = requests.post(f"{self.base_url}/api/generate", json={
                'prompt': prompt,
                'style': 'png',
                'quality': quality
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    # Decode base64 image
                    image_data = data['image'].split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    
                    return {
                        'success': True,
                        'image': image_bytes,
                        'generation_time': 0  # Will be calculated by caller
                    }
                else:
                    return {
                        'success': False,
                        'error': data.get('error', 'Unknown error')
                    }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_color_accuracy(self, prompt: str, image_bytes: bytes, expected_colors: List[str]) -> float:
        """Analyze color accuracy of generated image"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB
            image = image.convert('RGB')
            
            # Get dominant colors
            colors = image.getcolors(maxcolors=256*256*256)
            if not colors:
                return 0.0
            
            # Sort by frequency
            colors.sort(key=lambda x: x[0], reverse=True)
            
            # Convert to color names
            dominant_colors = []
            for count, (r, g, b) in colors[:10]:  # Top 10 colors
                color_name = self._rgb_to_color_name(r, g, b)
                if color_name:
                    dominant_colors.append(color_name)
            
            # Calculate accuracy
            matches = 0
            for expected_color in expected_colors:
                if expected_color in dominant_colors:
                    matches += 1
            
            return matches / len(expected_colors) if expected_colors else 0.0
            
        except Exception as e:
            print(f"Color analysis error: {e}")
            return 0.0
    
    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """Convert RGB to color name"""
        color_ranges = {
            'red': [(150, 0, 0), (255, 150, 150)],
            'blue': [(0, 0, 150), (150, 150, 255)],
            'green': [(0, 150, 0), (150, 255, 150)],
            'yellow': [(150, 150, 0), (255, 255, 150)],
            'purple': [(150, 0, 150), (255, 150, 255)],
            'orange': [(255, 165, 0), (255, 200, 100)],
            'pink': [(255, 192, 203), (255, 220, 220)],
            'brown': [(139, 69, 19), (200, 150, 100)],
            'white': [(200, 200, 200), (255, 255, 255)],
            'black': [(0, 0, 0), (100, 100, 100)]
        }
        
        for color_name, (min_rgb, max_rgb) in color_ranges.items():
            if (min_rgb[0] <= r <= max_rgb[0] and 
                min_rgb[1] <= g <= max_rgb[1] and 
                min_rgb[2] <= b <= max_rgb[2]):
                return color_name
        
        return None
    
    def _save_image(self, image_bytes: bytes, filename: str):
        """Save image to file"""
        try:
            with open(filename, 'wb') as f:
                f.write(image_bytes)
        except Exception as e:
            print(f"Error saving image {filename}: {e}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        # Color accuracy results
        color_tests = [r for r in self.test_results if 'color_accuracy' in r]
        if color_tests:
            avg_accuracy = sum(r['color_accuracy'] for r in color_tests) / len(color_tests)
            print(f"\n🎨 Color Accuracy: {avg_accuracy:.2f} ({avg_accuracy*100:.1f}%)")
            
            for test in color_tests:
                status = "✅" if test['color_accuracy'] > 0.7 else "⚠️" if test['color_accuracy'] > 0.4 else "❌"
                print(f"  {status} {test['test']}: {test['color_accuracy']:.2f}")
        
        # Performance results
        perf_tests = [r for r in self.test_results if 'quality' in r]
        if perf_tests:
            print(f"\n⚡ Performance Results:")
            for test in perf_tests:
                if test['success']:
                    status = "✅" if test['within_expected'] else "⚠️"
                    print(f"  {status} {test['quality']}: {test['actual_time']:.2f}s (Expected: {test['expected_time']}s)")
                else:
                    print(f"  ❌ {test['quality']}: Failed - {test.get('error', 'Unknown error')}")
        
        # Overall success rate
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get('success', False)])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📈 Overall Success Rate: {success_rate:.2f} ({success_rate*100:.1f}%)")
        print(f"✅ Successful Tests: {successful_tests}/{total_tests}")
        
        # Save detailed report
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'color_accuracy': avg_accuracy if color_tests else 0,
            'test_results': self.test_results
        }
        
        with open('test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: test_report.json")
        print("="*60)

def main():
    """Run comprehensive production system tests"""
    print("🚀 Giramille Production System Tester")
    print("="*60)
    
    tester = ProductionSystemTester()
    
    # Run all tests
    tester.test_system_health()
    tester.test_color_accuracy()
    tester.test_performance()
    
    # Generate report
    tester.generate_report()
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
>>>>>>> 93065687c720c01a1e099ca0338e62bd0fa3ae90
