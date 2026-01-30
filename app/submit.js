import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

export default function AppointmentSuccess() {
  const router = useRouter();
  const params = useLocalSearchParams();

  const {
    firstName,
    lastName,
    age,
    gender,
    reason,
    bookingFor,
    assignedDoctorName,
    appointmentDate,
    selectedTime,
  } = params;

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>Appointment Booked</Text>
        <Text style={styles.subHeaderText}>
          Your appointment has been successfully booked.
        </Text>
      </View>

      <View style={styles.card}>
        {/* Big Circle with Check Icon */}
        <View style={styles.iconWrapper}>
          <View style={styles.iconCircle}>
            <Ionicons name="checkmark" size={90} color="#fff" />
          </View>
          <Text style={styles.successText}>Success!</Text>
        </View>

        <Text style={styles.cardTitle}>Appointment Details</Text>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Doctor Assigned</Text>
          <Text style={styles.value}>{assignedDoctorName}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Appointment Date</Text>
          <Text style={styles.value}>{appointmentDate}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Time</Text>
          <Text style={styles.value}>{selectedTime}</Text>
        </View>

        <View style={styles.divider} />

        <View style={styles.detailRow}>
          <Text style={styles.label}>Booking For</Text>
          <Text style={styles.value}>
            {bookingFor === 'yourself' ? 'Yourself' : 'Another Person'}
          </Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Last Name</Text>
          <Text style={styles.value}>{lastName}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>First Name</Text>
          <Text style={styles.value}>{firstName}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Age</Text>
          <Text style={styles.value}>{age}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Gender</Text>
          <Text style={styles.value}>{gender}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Reason</Text>
          <Text style={styles.value}>{reason}</Text>
        </View>
      </View>

      <TouchableOpacity
        style={styles.doneButton}
        onPress={() => router.replace('/')}
      >
        <Text style={styles.doneButtonText}>Go to Dashboard</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  headerText: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#333',
  },
  subHeaderText: {
    fontSize: 16,
    color: '#666',
    marginTop: 5,
    textAlign: 'center',
  },
  card: {
    backgroundColor: '#f5f5f5',
    borderRadius: 15,
    padding: 25,
    marginBottom: 20,
  },
  iconWrapper: {
    alignItems: 'center',
    marginBottom: 20,
  },
  iconCircle: {
    width: 140,
    height: 140,
    borderRadius: 70,
    backgroundColor: '#99E0E9',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOpacity: 0.15,
    shadowOffset: { width: 0, height: 3 },
    shadowRadius: 5,
    elevation: 5,
  },
  successText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2260FF',
    marginTop: 5,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
    paddingBottom: 5,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  label: {
    fontWeight: 'bold',
    color: '#555',
  },
  value: {
    color: '#333',
    maxWidth: '60%',
    textAlign: 'right',
  },
  divider: {
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
    marginVertical: 15,
  },
  doneButton: {
    backgroundColor: '#1E88E5',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 10,
  },
  doneButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
