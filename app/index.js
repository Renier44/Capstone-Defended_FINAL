import 'expo-router/entry';
import { useRouter } from 'expo-router';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
  Image
} from 'react-native';
import { useFonts } from 'expo-font';
import { SafeAreaView } from 'react-native-safe-area-context';

// Load required fonts
const customFonts = {
  SmartFont: require('../assets/fonts/ArchivoBlack-Regular.ttf'),
  SightFont: require('../assets/fonts/VarelaRound-Regular.ttf'),
};

export default function HomeScreen() {
  const router = useRouter();
  const [fontsLoaded] = useFonts(customFonts);

  const handleLetsGo = () => {
    router.push('/login');
  };

  if (!fontsLoaded) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#FFD24D" />
        <Text style={styles.loadingText}>Loading SmartSight...</Text>
      </View>
    );
  }

  return (
    <View style={styles.fullScreenContainer}>
      <SafeAreaView style={styles.content}>

        {/* Heading */}
        <View style={styles.headingBlock}>

          {/* SMART text */}
          <Text style={styles.smartText}>SMART</Text>

          {/* SIGHT + ICON */}
          <View style={styles.sightRow}>
            <Text style={styles.sightText}>SIGHT</Text>
            <Image
              source={require('../assets/images/icon.png')}
              style={styles.logo}
            />
          </View>

        </View>

        {/* Removed the description here */}

        {/* Button */}
        <TouchableOpacity style={styles.button} onPress={handleLetsGo}>
          <Text style={styles.buttonText}>Let’s Go</Text>
        </TouchableOpacity>

      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  // Loading Page
  loadingContainer: {
    flex: 1,
    backgroundColor: '#FFFFF0',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#555',
    fontFamily: 'SightFont',
  },

  // Main Page
  fullScreenContainer: {
    flex: 1,
    backgroundColor: '#FFFFF0',
    justifyContent: 'center',
    alignItems: 'center',
  },

  content: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 20,
  },

  headingBlock: {
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 10,
  },

  smartText: {
    fontSize: 65,
    color: '#77CDE0',
    fontFamily: 'SmartFont',
    letterSpacing: 5,
    lineHeight: 90,
    textAlign: 'center',
    marginBottom: -10,
    marginTop: 40,
    marginRight: 30,
  },

  sightRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },

  sightText: {
    fontSize: 60,
    color: '#FFD24D',
    fontFamily: 'SightFont',
    letterSpacing: 5,
    lineHeight: 90,
    textAlign: 'center',
    marginRight: 5,
  
  },

  logo: {
    width: 105,
    height: 105,
    resizeMode: 'contain',
    marginLeft: -5,
  },

  button: {
    backgroundColor: '#77CDE0',
    paddingVertical: 14,
    paddingHorizontal: 60,
    borderRadius: 30,
    marginTop: 20,
    marginBottom: -5,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 3,
  },

  buttonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
    fontFamily: 'SightFont',
  },
});
