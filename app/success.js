// app/success.js
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
// IMPORT useLocalSearchParams for dynamic content
import { useRouter, useLocalSearchParams } from 'expo-router'; 
import { Ionicons } from '@expo/vector-icons';

export default function Success() {
  const router = useRouter();
  // Get success status and message from URL parameters
  const { success, message } = useLocalSearchParams(); 

  const isSuccess = success === 'true'; // Check if the booking was successful

  return (
    <View style={styles.container}>
      {isSuccess ? (
        <>
          {/* 1. Outline Checkmark (Larger, Light Blue) */}
          <Ionicons
            name="checkmark-outline"
            size={120} 
            color="#77CDE0" 
            style={styles.outlineCheck}
          />
          {/* 2. Solid Checkmark inside the main circle */}
          <View style={styles.circle}>
            <Ionicons name="checkmark" size={80} color="#FFFFFF" />
          </View>
          <Text style={styles.confirmationHeader}>Appointment Confirmed!</Text>
        </>
      ) : (
        <>
          {/* Fallback for failure/error */}
          <Ionicons name="close-circle" size={120} color="#FF6347" style={styles.errorIcon} />
          <Text style={styles.confirmationHeaderError}>Booking Failed</Text>
        </>
      )}

      {/* Dynamic message text */}
      <Text style={styles.messageText}>
        {message || 
         (isSuccess 
           ? 'Your appointment has been submitted successfully.' 
           : 'There was an issue processing your request. Please try again.')}
      </Text>

      {/* Button navigates back to the dashboard/home screen */}
      <TouchableOpacity 
        style={styles.button} 
        onPress={() => router.replace('/(tabs)/dashboard')}
      >
        <Text style={styles.buttonText}>Go to Dashboard</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  // --- Container (Matches schedule.js Background) ---
  container: {
    flex: 1,
    backgroundColor: '#C8EAF7', 
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },

  // --- Success Icon Layout (Two-Layer Checkmark) ---
  outlineCheck: {
    marginBottom: -40, // Pulls the outline checkmark closer to the solid circle
    alignSelf: 'center',
  },
  circle: {
    backgroundColor: '#77CDE0', // Requested full color
    borderRadius: 100,
    height: 160,
    width: 160,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 30, 
    // Consistent shadow from other UIs
    shadowColor: '#000',
    shadowOpacity: 0.15,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 6,
    elevation: 8,
  },
  errorIcon: {
    marginBottom: 40,
  },

  // --- Header Text ---
  confirmationHeader: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2260FF', // Consistent primary blue
    fontFamily: 'LeagueSpartan', // Consistent font
    marginBottom: 10,
  },
  confirmationHeaderError: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FF6347', // Error color
    fontFamily: 'LeagueSpartan',
    marginBottom: 10,
  },
  messageText: {
    fontSize: 18,
    textAlign: 'center',
    marginBottom: 40,
    color: '#333',
    fontFamily: 'LeagueSpartan', // Consistent font
    lineHeight: 25,
  },

  // --- Button (Matches schedule.js Button) ---
  button: {
    backgroundColor: '#FFD54F', // Consistent yellow button
    paddingVertical: 13,
    paddingHorizontal: 32,
    borderRadius: 25, // Consistent rounded button shape
    // Consistent shadow
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowOffset: { width: 0, height: 4 },
    shadowRadius: 6,
    elevation: 8,
  },
  buttonText: {
    color: '#fff', 
    fontWeight: 'bold',
    fontSize: 18, 
    fontFamily: 'LeagueSpartan', // Consistent font
  },
});