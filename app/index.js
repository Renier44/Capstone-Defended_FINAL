import 'expo-router/entry';
import { useRouter } from 'expo-router';
import { Image, SafeAreaView, StyleSheet, Text, TouchableOpacity, View, Dimensions } from 'react-native';

const { height } = Dimensions.get('window');

export default function HomeScreen() {
  const router = useRouter();

  const handleLetsGo = () => {
    router.push('/login'); // Navigate to login screen
  };

  return (
    <SafeAreaView style={styles.container} edges={['left', 'right', 'bottom']}>
      {/* Top Large Rounded Image Section */}
      <View style={styles.topImageContainer}>
        <Image
          source={require('../assets/images/home.png')}
          style={styles.topImage}
        />
      </View>

      <View style={styles.content}>
        {/* Heading + Logo side by side (Logo on right, Text on left) */}
        <View style={styles.headingRow}>
          {/* Heading Text - Now placed first */}
          <View style={styles.headingText}>
            <Text style={styles.smartText}>SMART</Text>
            <Text style={styles.sightText}>SIGHT</Text>
          </View>

          {/* Logo - Now placed second */}
          <Image
            source={require('../assets/images/icon.png')}
            style={styles.logo}
          />
        </View>

        <Text style={styles.bodyText}>
          Welcome to SmartSight! Your vision care journey with Enhance Vision Optical PH starts here.
        </Text>

        <TouchableOpacity style={styles.button} onPress={handleLetsGo}>
          <Text style={styles.buttonText}>Let’s Go</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#77CDE0',
  },
  topImageContainer: {
    width: '100%',
    height: height * 0.55,
    borderBottomLeftRadius: 60,
    borderBottomRightRadius: 60,
    overflow: 'hidden',
    backgroundColor: '#77CDE0',
  },
  topImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
    alignSelf: 'flex-start',
  },
  content: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 20,
    // Adjusted to 0 to move content right below the top image
    marginTop: 0, 
  },
  headingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    // Reduced marginBottom from 15 to 5 to pull the body text up closer to the heading
    marginBottom: 5,
  },
  headingText: {
    // Alignment set back to 'flex-start' to left-align SMART and SIGHT
    alignItems: 'flex-start',
    // Removed marginLeft to make text flush with logo
    marginLeft: 0,
    // Removed custom margin to allow vertical centering with the logo
    // marginTop: 10,
  },
  smartText: {
    // Increased font size for a thicker look
    fontSize: 60, 
    // Set to 700 (bold) and Varela Round for the desired thick, rounded look
    fontWeight: '700', 
    color: '#FFFFFF', 
    // Using VarelaRound which matches the image style (ensure it's loaded in your project)
    fontFamily: 'VarelaRound', 
    lineHeight: 62,
    // Added letter spacing to match the design style
    letterSpacing: 5, 
  },
  sightText: {
    // Increased font size for a thicker look
    fontSize: 60,
    // Set to 700 (bold) and Varela Round for the desired thick, rounded look
    fontWeight: '700',
    color: '#FFD24D',
    // Using VarelaRound which matches the image style (ensure it's loaded in your project)
    fontFamily: 'VarelaRound',
    lineHeight: 62,
    // Increased letter spacing to match the design style
    letterSpacing: 5, 
    // Removed margin to align SIGHT perfectly under SMART
    marginLeft: 0, 
  },
  logo: {
    // Updated size to 140x140 as requested
    width: 140, 
    height: 140,
    resizeMode: 'contain',
    // Adjusted margin to 20 for better vertical centering
    marginTop: 20, 
  },
  bodyText: {
    fontSize: 16,
    fontWeight: '300',
    color: '#FFFFFF',
    textAlign: 'center',
    lineHeight: 24,
    maxWidth: 320,
    // Reduced marginBottom from 20 to 10 to pull the button up
    marginBottom: 10,
    fontFamily: 'Roboto',
  },
  button: {
    backgroundColor: '#FFFFFF',
    paddingVertical: 14,
    paddingHorizontal: 60,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  buttonText: {
    color: '#65A3D5',
    fontSize: 18,
    fontWeight: '600',
    fontFamily: 'Roboto',
  },
});
