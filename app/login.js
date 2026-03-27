import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Image,
  ActivityIndicator,
  Platform, // << ADDED
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { useRouter } from 'expo-router';
import * as SecureStore from 'expo-secure-store';
import { Ionicons } from '@expo/vector-icons';
import { useFonts } from 'expo-font'; // << ADDED

const API_BASE_URL = 'https://capstone-defended-final.onrender.com/api';


export default function Login() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  // Load required custom fonts for the brand name (SMART/SIGHT)
  const [fontsLoaded] = useFonts({
    SmartFont: require('../assets/fonts/ArchivoBlack-Regular.ttf'),
    SightFont: require('../assets/fonts/VarelaRound-Regular.ttf'),
  });

  if (!fontsLoaded) return null;


  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Missing Fields', 'Please enter both email and password.');
      return;
    }

    setLoading(true);

    try {
      const payload = {
        username: email.trim(),
        password,
      };

      const response = await fetch(`${API_BASE_URL}/login/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server responded with status:', response.status);
        console.error('Raw error content:', errorText.substring(0, 300) + '...');
        let errorMessage = 'Login failed. Please try again.';
        try {
          const parsedError = JSON.parse(errorText);
          if (parsedError.message) errorMessage = parsedError.message;
        } catch {}
        if (response.status === 403 && errorMessage.includes('blocked')) {
          throw new Error('Your account has been blocked by the admin.');
        } else if (response.status === 401) {
          throw new Error('Invalid email or password. Please try again.');
        } else if (response.status === 400) {
          throw new Error('Authentication failed. Please confirm your email/password.');
        } else {
          throw new Error(errorMessage);
        }
      }

      const data = await response.json();
      console.log('Login response data:', data);

      if (data.token) {
        // Save token
        await SecureStore.setItemAsync('userToken', data.token);

        // existingProfile merging (if any) preserved
        const existingProfileStr = await SecureStore.getItemAsync('userProfile');
        const existingProfile = existingProfileStr ? JSON.parse(existingProfileStr) : {};

        const finalProfileToSave = {
          // Take existing fields (like old profile_image) but prioritize new data from server
          ...existingProfile, 
          ...data, // This line merges the new data (including date_of_birth, gender, first_name, last_name, email)
          
          // Explicitly ensure the profile image is kept if the login response didn't include it
          profile_image: existingProfile.profile_image || data.profile_image, 
          
          // Ensure the keys match what EditProfile.js expects:
          first_name: data.first_name, 
          last_name: data.last_name,
          email: data.email,
          gender: data.gender,
          date_of_birth: data.date_of_birth,
        };

        await SecureStore.setItemAsync('userProfile', JSON.stringify(finalProfileToSave));

        // ALSO save a lightweight 'user' object used by EyeScreening (id, username, email)
        const lightweightUser = {
          id: data.id || data.user_id || (data.user ? data.user.id : undefined),
          username: data.username || (data.user ? data.user.username : undefined),
          email: data.email || (data.user ? data.user.email : undefined),
        };
        // remove undefined keys:
        Object.keys(lightweightUser).forEach(k => lightweightUser[k] === undefined && delete lightweightUser[k]);

        if (Object.keys(lightweightUser).length > 0) {
          await SecureStore.setItemAsync('user', JSON.stringify(lightweightUser));
        }

        // Success, redirect
        Alert.alert('Login Successful', 'Welcome to SmartSight!', [
          {
            text: 'Continue',
            onPress: () => router.replace('/dashboard'),
          },
        ]);
      } else {
        Alert.alert('Login Failed', data.message || 'Invalid credentials or missing token.');
      }
    } catch (error) {
      console.error('Login error:', error);
      Alert.alert('Login Error', error.message || 'Failed to connect to the server. Check your network.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#FFFFFF' }}> 
      <View style={styles.container}>
        <View style={styles.brandRow}>
          <Text style={styles.appName}>
            {/* START: Changed to stacked layout */}
            <Text style={styles.smart}>SMART{'\n'}</Text>
            <Text style={styles.sight}>SIGHT</Text>
            {/* END: Changed to stacked layout */}
          </Text>
          <Image
            source={require('../assets/images/icon.png')}
            style={styles.logo}
          />
        </View>

        {/* Added subBrand text */}
        <Text style={styles.subBrand}>Enhance Vision Optical PH</Text>

        <Text style={styles.instruction}>sign in to access your account</Text>

        <View style={styles.inputContainer}>
          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.input}
              placeholder="Enter your email"
              placeholderTextColor="#666"
              autoCapitalize="none"
              keyboardType="email-address"
              value={email}
              onChangeText={setEmail}
              editable={!loading}
            />
            <Ionicons name="mail-outline" size={20} color="#666" />
          </View>

          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.input}
              placeholder="Password"
              placeholderTextColor="#666"
              secureTextEntry={!showPassword}
              value={password}
              onChangeText={setPassword}
              editable={!loading}
            />
            <Ionicons name="lock-closed-outline" size={20} color="#666" />
          </View>

          <TouchableOpacity onPress={() => setShowPassword(!showPassword)} disabled={loading}>
            <Text style={styles.showPassword}>
              {showPassword ? 'Hide password' : 'Show password'}
            </Text>
          </TouchableOpacity>
        </View>

        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#FFD54F" />
            {/* Note: loadingText color was fixed to be visible on white background */}
            <Text style={styles.loadingText}>Logging in...</Text>
          </View>
        ) : (
          <TouchableOpacity style={styles.button} onPress={handleLogin}>
            <Text style={styles.buttonText}>Login</Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity disabled={loading} onPress={() => router.push('/register')}>
          <Text style={styles.register}>
            New Member? <Text style={styles.registerNow}>Register now</Text>
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}


// ====================
// ✅ UPDATED STYLES with custom font families
// ====================
const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    backgroundColor: '#FFFFFF', 
    padding: 30, 
    justifyContent: 'center' 
  },
  brandRow: { 
    flexDirection: 'row', 
    alignItems: 'center', 
    justifyContent: 'center', 
    marginBottom: 8 
  },
  appName: { 
    fontSize: 40, 
    // Removed specific fontWeight here as it is handled by the custom fonts
    textAlign: 'right', 
    lineHeight: Platform.OS === 'ios' ? 44 : 42, 
    textAlignVertical: 'center', 
    includeFontPadding: false, 
    marginRight: 15 
  },
  smart: { 
    color: '#77CDE0',
    fontFamily: 'SmartFont', // <<< Applied ArchivoBlack-Regular
  },
  sight: { 
    color: '#FFD54F', 
    fontFamily: 'SightFont' // <<< Applied VarelaRound-Regular
    
  },
  logo: { 
    width: 100, 
    height: 100, 
    resizeMode: 'contain' 
  },
  subBrand: { 
    fontSize: 14, 
    color: '#999', 
    textAlign: 'center', 
    marginBottom: 20 
  },
  instruction: { // Text outside the box
    textAlign: 'center',
    color: '#555',
    marginBottom: 25,
    fontSize: 15,
  },
  inputContainer: {
    backgroundColor: '#99E0E9', 
    borderRadius: 20,
    padding: 15,
    marginBottom: 20,
    // Added shadow/elevation from previous code
    ...Platform.select({
      ios: { shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.15, shadowRadius: 3 },
      android: { elevation: 2 },
    }),
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#C8EAF7',
    borderRadius: 12,
    marginBottom: 15,
    paddingHorizontal: 15,
    height: 50,
  },
  input: { flex: 1, fontSize: 16 },
  showPassword: {
    textAlign: 'right',
    color: '#5B4FE9',
    fontSize: 13,
    marginTop: 5,
  },
  button: {
    backgroundColor: '#FFD54F',
    borderRadius: 25,
    height: 50,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
    // Added button shadow/elevation from previous code
    ...Platform.select({ 
      ios: { shadowColor: '#000', shadowOffset: { width: 0, height: 3 }, shadowOpacity: 0.2, shadowRadius: 4 }, 
      android: { elevation: 3 } 
    })
  },
  buttonText: { color: '#fff', fontWeight: 'bold', fontSize: 18 },
  register: { textAlign: 'center', color: '#555', fontSize: 14 },
  registerNow: { color: '#5B4FE9', fontWeight: '600' },

  // Spinner styles 
  loadingContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
    gap: 10,
  },
  loadingText: {
    color: '#555', 
    fontWeight: 'bold',
    fontSize: 16,
    marginLeft: 10,
  },
});